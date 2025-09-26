import torch
from tqdm import tqdm
from jiwer import wer
from ctcdecode import CTCBeamDecoder
from losses import EOSEnhancedCrossEntropyLoss, FocalLoss

import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from ctcdecode import CTCBeamDecoder
from jiwer import wer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from super_monotonic_align import maximum_path
from batchdilate import DTWShpTime
from torch.nn.utils.rnn import pad_sequence


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, tokenizer, device,
                 ctc_loss_weight=0.3, ce_loss_weight=0.7, duration_loss_weight=0.0,
                   max_norm=1.0, accumulation_target=128, mamba_enc=True):
        """
        model: ASR 模型
        optimizer: 優化器
        train_loader: 訓練資料 DataLoader，回傳 (spectrograms, labels, input_lengths, label_lengths)
        valid_loader: 驗證資料 DataLoader，格式同上
        tokenizer: SentencePieceTransform 物件，用於 token 與文字轉換
        device: 運算設備 ('cpu' 或 'cuda')
        ctc_loss_weight, ce_loss_weight: 兩個 loss 的加權比例
        blank_id: CTC 中 blank token 的 id (預設為 0)
        pad_id: AED loss 中填充符號的 id (用於 ignore_index)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.tokenizer = tokenizer
        self.device = device
        self.ctc_loss_weight = ctc_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.duration_loss_weight = duration_loss_weight  # 如果有使用 duration loss
        self.blank_id = tokenizer.sp.piece_to_id("<blank>")
        self.pad_id = tokenizer.sp.piece_to_id("<pad>")
        self.sos_id = tokenizer.sp.bos_id()
        self.eos_id = tokenizer.sp.eos_id()
        print("blank_id:", self.blank_id)
        print("pad_id:", self.pad_id)
        print("sos_id:", self.sos_id)
        print("eos_id:", self.eos_id)
        self.max_norm = max_norm
        self.accumulation_target = accumulation_target
        self.mamba_enc = mamba_enc
        self.use_dur = True if self.duration_loss_weight > 0 else False
        self.ctc_decoder = CTCBeamDecoder(
                    # [''] * (1024 - 1) + [' '],
                    labels = [tokenizer.sp.IdToPiece(i) for i in range(tokenizer.sp.GetPieceSize())],
                    model_path=None,
                    alpha=0,
                    beta=0,
                    cutoff_top_n=40,
                    cutoff_prob=1.0,
                    beam_width=1,
                    num_processes=4,
                    blank_id=self.blank_id,
                    log_probs_input=True
                )


        # 定義 loss 函數：CTC Loss 與 AED Loss (交叉熵)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        # self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id,
        #                                     # label_smoothing=0.1,
        #                                     reduction='mean')
        self.ce_loss_fn = EOSEnhancedCrossEntropyLoss(
            ignore_index=self.pad_id,
            # label_smoothing=0.1,
            eos_id=self.eos_id,
            eos_weight=1.5
        )

        # self.dur_loss_fn = DTWShpTime(alpha=0.8, gamma=0.01, reduction="mean")
        self.dur_loss_fn = Focal_loss(
            alpha=None,  # None 表示不使用 alpha 平衡
            gamma=2.0,  # 聚焦係數
            reduction='mean',  # 最終 loss 的平均
        )
        # self.dur_loss_fn = nn.CrossEntropyLoss(
        #     # label_smoothing=0.1,  # 可以調整這個值
        #     reduction='mean'
        # )

        self.pre_lr = self.optimizer.param_groups[0]['lr']

        self.train_loss_history = []
        self.train_ctc_loss_history = []
        self.train_ce_loss_history = []
        self.train_dur_loss_history = []

        self.valid_loss_history = []
        self.valid_ctc_loss_history = []
        self.valid_ce_loss_history = []
        self.valid_dur_loss_history = []

        self.valid_ctc_wer_history = []
        self.valid_aed_wer_history = []
    
    def train(self, epoch):
        self.model.train()
        total_ctc_loss = 0.0
        total_ce_loss = 0.0
        total_dur_loss = 0.0
        total_loss = 0.0
        running_samples = 0
        skip_count = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            spectrograms, labels, ar_tgts, aed_labels, input_lengths, label_lengths = batch

            src = spectrograms.squeeze(1).transpose(1, 2).to(self.device)
            labels = labels.to(self.device)
            aed_labels = aed_labels.to(self.device)
            ar_tgts = ar_tgts.to(self.device)

            current_batch_size = src.size(0)
            input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long, device=self.device)
            label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long, device=self.device)

            with autocast(device_type="cuda"):
                if self.ce_loss_weight > 0:
                    ctc_logits, ar_logits, ctc_output_lengths = self.model(src, ar_tgts, input_lengths_tensor, label_lengths_tensor)
                    # ar_logits[..., self.blank_id] = -6.5e4
                else:
                    ctc_logits, ctc_output_lengths = self.model(src, input_lengths_tensor)

                if self.mamba_enc:
                    ctc_logits = self.left_to_right_padding(ctc_logits, ctc_output_lengths)

                if self.use_dur:
                    dur_logits = ar_logits[:, :, -5:]  # 假設最後5個維度是 duration logits
                    ar_logits = ar_logits[:, :, :-5]  # 去掉 duration logits
                
                # 計算CTC Loss
                loss_ctc = self.cal_ctc_loss(ctc_logits, labels, ctc_output_lengths, label_lengths)

                # 計算AR Loss
                if self.ce_loss_weight > 0:
                    loss_ce = self.cal_ce_loss(ar_logits, aed_labels)
                else:
                    loss_ce = torch.tensor(0.0).to(self.device)

                # 計算Duration Loss
                if self.dur_loss_weight > 0:
                    loss_dur = self.cal_dur_loss(ctc_logits, ar_logits, dur_logits, ctc_output_lengths, label_lengths_tensor)
                else:
                    loss_dur = torch.tensor(0.0).to(self.device)

                # 累積loss
                loss = (self.ctc_loss_weight * loss_ctc + self.ce_loss_weight * loss_ce + self.dur_loss_weight * loss_dur) * (current_batch_size / self.accumulation_target)

                if not torch.isfinite(loss):
                    new_scale = max(self.scaler.get_scale() / 2, 1.0)

                    del self.scaler
                    torch.cuda.empty_cache()

                    # 重新初始化一個新的 GradScaler
                    self.scaler = torch.amp.GradScaler(init_scale=new_scale)

                    self.optimizer.zero_grad(set_to_none=True)
                    skip_count += 1
                    running_samples = 0
                    continue

                total_ctc_loss += loss_ctc.item()

                if self.ce_loss_weight > 0:
                    total_ce_loss += loss_ce.item()

                if self.dur_loss_weight > 0:
                    total_dur_loss += loss_dur.item()

                total_loss += (self.ctc_loss_weight * loss_ctc + self.ce_loss_weight * loss_ce + self.dur_loss_weight * loss_dur).item()

                self.scaler.scale(loss).backward()
                running_samples += current_batch_size

                # 當累積的樣本數達到目標，進行參數更新
                if running_samples >= self.accumulation_target:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()  # 更新學習率
                    self.optimizer.zero_grad()
                    running_samples = 0

                pbar.set_postfix({'loss': (self.ctc_loss_weight * loss_ctc + self.ce_loss_weight * loss_ce).item(), 'CTC_loss': loss_ctc.item(), 'CE_loss': loss_ce.item(), 'Dur_loss': loss_dur.item() if self.dur_loss_weight > 0 else 0.0, 'lr': self.optimizer.param_groups[0]['lr']})

        # 若最後累積的樣本數不足，仍需要更新一次
        if running_samples > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # 更新學習率
            self.optimizer.zero_grad()
            
        num_batches = len(self.train_loader)
        avg_ctc_loss = total_ctc_loss / (num_batches - skip_count)
        avg_ce_loss = total_ce_loss / (num_batches - skip_count) if self.ce_loss_weight > 0 else 0.0
        avg_dur_loss = total_dur_loss / (num_batches - skip_count) if self.dur_loss_weight > 0 else 0.0
        avg_loss = total_loss / num_batches

        if skip_count > 0:
            print(f"Skipped {skip_count} batches due to NaN/Inf loss")
        
        return avg_loss, avg_ctc_loss, avg_ce_loss, avg_dur_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_ce_loss = 0.0
        total_dur_loss = 0.0
        n_batches = len(self.valid_loader)
        skip_count = 0
        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f"Valid Epoch {epoch}")
            for batch in pbar:
                spectrograms, labels, ar_tgts, aed_labels, input_lengths, label_lengths = batch
                src = spectrograms.squeeze(1).transpose(1, 2).to(self.device)
                ar_tgts = ar_tgts.to(self.device)  # [batch, tgt_seq_len]
                labels = labels.to(self.device)  # [batch, tgt_seq_len]
                aed_labels = aed_labels.to(self.device)  # [batch, tgt_seq_len]

                input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long, device=self.device)
                label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long, device=self.device)

                with autocast(device_type="cuda"):
                    if self.ce_loss_weight > 0:
                        ctc_logits, ar_logits, ctc_output_lengths = self.model(src, ar_tgts, input_lengths_tensor, label_lengths_tensor)
                        ar_logits[..., self.blank_id] = -6.5e4
                    else:
                        ctc_logits, ctc_output_lengths = self.model(src, input_lengths_tensor)

                    if self.mamba_enc:
                        ctc_logits = self.left_to_right_padding(ctc_logits, ctc_output_lengths)

                    if self.use_dur:
                        dur_logits = ar_logits[:, :, -5:]  # 假設最後5個維度是 duration logits
                        ar_logits = ar_logits[:, :, :-5]  # 去掉 duration logits

                    # CTC Loss
                    loss_ctc = self.cal_ctc_loss(ctc_logits, labels, ctc_output_lengths, label_lengths)

                    # AR Loss
                    if self.ce_loss_weight > 0:
                        loss_ce = self.cal_ce_loss(ar_logits, aed_labels)
                    else:
                        loss_ce = torch.tensor(0.0).to(self.device)

                    # Duration Loss
                    if self.dur_loss_weight > 0:
                        loss_dur = self.cal_dur_loss(ctc_logits, ar_logits, dur_logits, ctc_output_lengths, label_lengths_tensor)
                    else:
                        loss_dur = torch.tensor(0.0).to(self.device)

                    loss = self.ctc_loss_weight * loss_ctc + self.ce_loss_weight * loss_ce + self.dur_loss_weight * loss_dur
                    if not torch.isfinite(loss):
                        skip_count += 1
                        continue

                    total_ctc_loss += loss_ctc.item()
                    total_ce_loss += loss_ce.item()
                    if self.dur_loss_weight > 0:
                        total_dur_loss += loss_dur.item()

                    total_loss += loss.item()

                pbar.set_postfix({'loss': (self.ctc_loss_weight * loss_ctc + self.ce_loss_weight * loss_ce).item(), 'CTC_loss': loss_ctc.item(), 'CE_loss': loss_ce.item(), 'Dur_loss': loss_dur.item() if self.dur_loss_weight > 0 else 0.0})

            # 計算平均 loss
            avg_ctc_loss = total_ctc_loss / (n_batches - skip_count)
            avg_ce_loss = total_ce_loss / (n_batches - skip_count)
            avg_dur_loss = total_dur_loss / (n_batches - skip_count) if self.dur_loss_weight > 0 else 0.0
            avg_loss = total_loss / (n_batches - skip_count)
            
            if skip_count > 0:
                print(f"Skipped {skip_count} batches due to NaN/Inf loss")
                
            # self.scheduler.step(avg_loss)

            # self.print_predict(epoch, ar_logits, ctc_logits, ctc_output_lengths, labels, label_lengths)

            return avg_loss, avg_ctc_loss, avg_ce_loss, avg_dur_loss

    def test(self, test_loader):
        torch.cuda.empty_cache()
        self.model.eval()        
        all_targets=[]
        all_ctc_preds=[]
        # all_aed_preds=[]
        all_ar_preds=[]
        # all_ctc_dur=[]

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Test")
            for batch in pbar:
                spectrograms, labels, ar_tgts, aed_labels, input_lengths, label_lengths = batch

                # 將輸入轉成模型要求的 shape 與 device
                src = spectrograms.squeeze(1).transpose(1, 2).to(self.device)
                gt = labels.to(self.device)
                input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long, device=self.device)

                # 模型輸出為 token 序列，不是 logits
                if self.ce_loss_weight > 0:
                    ar_predictions, ctc_logits, ctc_lengths = self.model.fast_generate(src, input_lengths_tensor,
                                                                                        beam_size=5,
                                                                                        ctc_weight=self.ctc_loss_weight) # aed_predictions: [batch, max_len_out]
                else:
                    ctc_logits, ctc_lengths = self.model(src, input_lengths_tensor)
                    
                # 創建右填充版本
                if self.mamba_enc:
                    ctc_logits = self.left_to_right_padding(ctc_logits, ctc_lengths) # to right padding

                ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
                # print("ctc_log_probs:", ctc_log_probs.shape)

                # preds = ctc_logits.argmax(dim=-1)             # -> [B, T]
                # print("preds:", preds[0])
                # non_blank = (preds != self.blank_id)               # -> [B, T]

                # # 方法 B：torch 索引
                # ctc_dur_predictions = []
                # for b in range(preds.size(0)):
                #     seq_b = torch.masked_select(preds[b], non_blank[b])
                #     ctc_dur_predictions.append(seq_b.tolist())

                # 計算 CTC 分支 WER
                beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(ctc_log_probs, ctc_lengths)

                beam_results = beam_results.tolist()

                itera = spectrograms.size()
                b=[]
                for i in range(itera[0]):
                    b.append(beam_results[i][0][:out_lens[i][0]])
                ctc_predictions = b
                # print("ctc_predictions:", ctc_predictions[0])

                # 計算 AR 分支
                # print("ar predictions:", ar_predictions[0])

                # 轉換 ground-truth 序列（過濾掉 padding token）
                gt_texts = []
                gt_cpu = gt.cpu().tolist()
                for seq, l in zip(gt_cpu, label_lengths):
                    seq = [token for token in seq[:l] if token != self.pad_id]
                    
                    gt_texts.append(self.tokenizer.int_to_text(seq))

                # 轉換模型推理輸出的 token 序列（過濾掉 padding token）
                ctc_pred_texts = []
                for seq in ctc_predictions:
                    seq = [token for token in seq if token != self.pad_id]
                    ctc_pred_texts.append(self.tokenizer.int_to_text(seq))

                if self.ce_loss_weight > 0:
                    ar_pred_texts = []
                    for seq in ar_predictions:
                        seq = [tok for tok in seq if tok != self.pad_id]
                        ar_pred_texts.append(self.tokenizer.int_to_text(seq))

                # aed_pred_texts = []
                # for seq in aed_predictions:
                #     # 1. 找到第一個 <eos> 的位置
                #     try:
                #         eos_index = seq.index(self.eos_id)
                #         # 2. 截斷序列，只保留 <eos> 之前的部分
                #         seq = seq[:eos_index]
                #     except ValueError:
                #         # 如果序列中沒有 <eos>，則使用整個序列
                #         pass
                    
                #     # 3. 過濾掉可能存在的 padding token
                #     seq = [token for token in seq if token != self.pad_id]
                #     aed_pred_texts.append(self.tokenizer.int_to_text(seq))

                # ctc_dur_texts = []
                # for seq in ctc_dur_predictions:
                #     seq = [token for token in seq if token != self.pad_id]
                #     ctc_dur_texts.append(self.tokenizer.int_to_text(seq))

                # for ctc_dur in ctc_dur_predictions:
                #     all_ctc_dur.append(ctc_dur)

                # 計算 WER
                if self.ce_loss_weight > 0:
                    for gt, ctc_pred, ar_pred in zip(gt_texts, ctc_pred_texts, ar_pred_texts):
                        all_targets.append(gt)
                        all_ctc_preds.append(ctc_pred)
                        if self.ce_loss_weight > 0:
                            all_ar_preds.append(ar_pred)
                else:
                    for gt, ctc_pred in zip(gt_texts, ctc_pred_texts):
                        all_targets.append(gt)
                        all_ctc_preds.append(ctc_pred)

            # 使用換行符連接句子來建立語料庫，這通常比用空格更穩健
            corpus_ctc_wer = wer("\n".join(all_targets), "\n".join(all_ctc_preds)) * 100

            if self.ce_loss_weight > 0:
                corpus_ar_wer = wer("\n".join(all_targets), "\n".join(all_ar_preds)) * 100
                print(f"CTC WER: {corpus_ctc_wer:.2f}, AR WER: {corpus_ar_wer:.2f}")
            else:
                print(f"CTC WER: {corpus_ctc_wer:.2f}")    

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # for i in range(3):
            #     print(f"Sample {i + 1}:")
            #     print(f"Label: {gt_texts[i]}")
            #     print(f"CTC: {ctc_pred_texts[i]}")
            #     print(f"AR: {ar_pred_texts[i]}")
            
            '''
            # ---------------------------
            # vocab_size = 1024
            # max_repeats = compute_max_consecutive_repeats(all_ctc_dur, vocab_size)

            # # 1. 构造 (token_id, count) 列表
            # token_counts = [(token_id, max_repeats[token_id]) for token_id in range(vocab_size)]

            # # 2. 按 count 降序排序
            # token_counts.sort(key=lambda x: x[1], reverse=True)

            # # 3. 取前 100
            # top100 = token_counts[:100]

            # # 4. 打印出来
            # print(f"{'Rank':>4s} | {'Token ID':>8s} | {'Text':>10s} | {'Max Repeat':>10s}")
            # print("-" * 44)
            # for rank, (tid, cnt) in enumerate(top100, 1):
            #     # 如果想把 token id 转成文字，可以：
            #     txt = self.tokenizer.int_to_text([tid])
            #     print(f"{rank:4d} | {tid:8d} | {txt:10s} | {cnt:10d}")

            # # ---------------------------
            # # 新增：统计不同连续重复次数（duration）的 token 数量分布
            # # ---------------------------

            # # 5. 统计每个连续重复次数（duration）出现了多少个 token
            # #    先找到最大可能的连续重复次数（例如 N = max(max_repeats)）
            # max_duration = max(max_repeats)
            # # 建立一个字典：duration_count[d] = 出现连续重复次数等于 d 的 token 数量
            # duration_count = {d: 0 for d in range(1, max_duration+1)}  # 从 1 开始，因为 0 连续重复没有意义
            # for tid in range(vocab_size):
            #     d = max_repeats[tid]
            #     if d > 0:
            #         duration_count[d] += 1

            # # 6. 将 duration_count 转为列表，方便打印和绘图
            # durations = sorted(duration_count.keys())                # [1, 2, 3, …, max_duration]
            # counts_per_duration = [duration_count[d] for d in durations]

            # print("\n不同连续重复次数（duration）对应的 token 数量：")
            # print(f"{'Duration':>8s} | {'#Tokens':>8s}")
            # print("-" * 22)
            # for d, cnt in zip(durations, counts_per_duration):
            #     print(f"{d:8d} | {cnt:8d}")

            # # 7. 绘制直方图：横轴为连续重复次数（duration），纵轴为对应的 token 数量
            # plt.figure(figsize=(10, 4))
            # plt.bar(durations, counts_per_duration, width=0.8)
            # plt.xlabel("Consecutive Repeat Length (Duration)")
            # plt.ylabel("Number of Tokens")
            # plt.title("Distribution of Max Consecutive Repeats Across All Tokens")
            # plt.xticks(durations)  # 为了直观显示每个可能的重复次数
            # plt.tight_layout()
            # plt.show()

            # # 8. 原先的 token-level 柱状图（显示每个 token 的最大重复次数），保持不变
            # plt.figure(figsize=(12, 4))
            # plt.bar(range(vocab_size), max_repeats)
            # plt.xlabel("Token ID")
            # plt.ylabel("Max Consecutive Repeat Count")
            # plt.title("Max Consecutive Repeats of CTC Predictions")
            # plt.tight_layout()
            # plt.show()
            '''

            if self.ce_loss_weight > 0:
                return corpus_ctc_wer, corpus_ar_wer
            else:
                return corpus_ctc_wer

    def cal_ctc_loss(self, ctc_logits, labels, ctc_lengths, label_lengths):
        ctc_logits = ctc_logits.to(torch.float32)
        ctc_logits_t = ctc_logits.transpose(0, 1)  # [seq_len, batch, vocab_size]
        log_probs = F.log_softmax(ctc_logits_t, dim=-1).float()

        label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long, device=self.device)

        targets_list = []
        for i, l in enumerate(label_lengths):
            targets_list.append(labels[i, :l])
        targets_concat = torch.cat(targets_list)

        loss_ctc = self.ctc_loss_fn(log_probs, targets_concat, ctc_lengths, label_lengths_tensor)

        return loss_ctc
    
    def cal_ce_loss(self, ar_logits, ar_labels):
        ar_logits = ar_logits.to(torch.float32)
        ar_logits_flat = ar_logits.reshape(-1, ar_logits.size(-1))
        ar_targets_flat = ar_labels.reshape(-1)

        # ar_logits_predicted_ids = ar_logits.argmax(dim=-1)  # [B*T, V] -> [B*T]

        # ar_logits_flat_probs = F.softmax(ar_logits_flat, dim=-1)       # [B, T, V]
        # ar_logits_flat_predicted_ids = ar_logits_flat_probs.argmax(dim=-1)

        loss_ce = self.ce_loss_fn(ar_logits_flat, ar_targets_flat)  # 平均 loss

        # print("loss: ", loss_ce)
        # for i in range(batch_size):
        #     print("ar_logits_predicted_ids:")
        #     print(ar_logits_predicted_ids[i])
        #     print("ar_labels:")
        #     print(ar_labels[i])      

        return loss_ce
    
    def cal_dur_loss(self,
        ctc_logits:   torch.Tensor,   # [B, T_x, V]
        ar_logits:   torch.Tensor,   # [B, T_y, V]
        dur_logits:   torch.Tensor,   # [B, T_y, D]
        ctc_lengths:  torch.Tensor,   # [B]
        aed_lengths:  torch.Tensor) -> torch.Tensor:  # 回傳 loss
        blank_id = self.blank_id
        temp     = 0.8  # temperature for softmax
        device   = ctc_logits.device

        B, T_x, V = ctc_logits.size()
        _, T_y, D = dur_logits.size()

        # 1) CTC → log‐probs
        logp_ctc  = F.log_softmax(ctc_logits / temp, dim=-1)   # [B, T_x, V]
        preds_ctc = ctc_logits.argmax(dim=-1)                  # [B, T_x]

        # 2) 取出非‐blank frames
        mask_nonblank = (preds_ctc != blank_id)                # [B, T_x]
        #    篩後每個 batch 的有效長度
        ctc_len_filt  = mask_nonblank.sum(dim=1)               # [B]
        max_ctc_filt  = ctc_len_filt.max().item()

        # 3) 建立 ctc_logp_filt: [B, T_x', V]，其中 T_x' = max_ctc_filt
        ctc_logp_filt = torch.zeros((B, max_ctc_filt, V),
                                    device=device,
                                    dtype=logp_ctc.dtype)
        for b in range(B):
            idx_non = torch.nonzero(mask_nonblank[b]).squeeze(1)   # 取出所有非‐blank 的 x index
            logp_b  = logp_ctc[b, idx_non, :]                     # [ctc_len_filt[b], V]
            pad_len = max_ctc_filt - logp_b.size(0)
            if pad_len > 0:
                # 將剩餘 frame 填上極大負值，避免 DP 選到它們
                filler = torch.full((pad_len, V), -1e9,
                                     dtype=logp_ctc.dtype,
                                     device=device)
                logp_b  = torch.cat([logp_b, filler], dim=0)     # [T_x', V]
            ctc_logp_filt[b] = logp_b

        #    對應的篩後 CTC 長度
        ctc_lengths_filt = ctc_len_filt                        # [B]

        # 4) AED 取最可能 token
        preds_aed = ar_logits.argmax(dim=-1)                  # [B, T_y]
        # preds_dur = dur_logits.argmax(dim=-1)                  # [B, T_y] (debug 用)

        # 5) 找出 EOS 及其後（不監督）
        is_eos   = preds_aed == self.eos_id                    # [B, T_y]
        eos_pos  = is_eos.float().cumsum(dim=1)
        eos_mask = eos_pos > 0                                 # [B, T_y]
        #    將 EOS 以後全部標為 pad_id
        preds_aed = preds_aed.masked_fill(eos_mask, self.pad_id)  # [B, T_y]

        # 6) MAS input：value, mask 維度要是 [B, T_y, T_x']
        value, mask = self.prepare_mas_input(
            log_probs     = ctc_logp_filt,           # [B, T_x', V]
            labels        = preds_aed,               # [B, T_y]
            input_lengths = ctc_lengths_filt,        # [B]
            label_lengths = aed_lengths,             # [B]
            padding_id    = blank_id
        )
        #    此時：value.shape == [B, T_y, T_x']，mask.shape == [B, T_y, T_x']

        # Debug prints（可視化前幾個值與 mask）
        # print("value ([B,T_y,T_x′]) sample:", value[0, :T_y, :min(5, ctc_lengths_filt[0].item())])
        # print("mask  ( [B,T_y,T_x′]) sample:", mask[0, :T_y, :min(5, ctc_lengths_filt[0].item())])

        # 7) 動態規劃 → path:[B, T_y, T_x']
        path       = maximum_path(value, mask, dtype=torch.bool)  # [B, T_y, T_x']
        dur_targets = path.sum(dim=2).long()                      # [B, T_y]

        # 8) 建立 supervision mask
        is_eos    = preds_aed == self.eos_id                      # [B, T_y]
        eos_pos   = is_eos.float().cumsum(dim=1)
        eos_mask  = eos_pos > 0                                   # [B, T_y]

        idx       = torch.arange(T_y, device=device)[None, :].expand(B, -1)  # [B, T_y]
        seq_mask  = idx < aed_lengths[:, None]                              # [B, T_y]
        sup_mask  = (~eos_mask) & seq_mask                                   # [B, T_y]

        # 9) 將 logits 與 targets 攤平成一維
        B_Ty = B * T_y
        D    = dur_logits.size(-1)                   # 最大類別數（duration 的最大值+1）
        flat_logits  = dur_logits.reshape(B_Ty, D)   # [B*T_y, D]
        flat_targets = dur_targets.reshape(B_Ty)     # [B*T_y]
        flat_sup     = sup_mask.reshape(B_Ty)        # [B*T_y]

        # 10) 把所有大於 (D-1) 的目標值都 clamp 成 D-1
        #     這樣交叉熵的 targets_sel 就一定落在 [0, D-1] 範圍內
        clamped_targets = torch.clamp(flat_targets, max=D-1)  # [B*T_y]

        # 11) 只保留 supervsion mask（把 EOS 之後和 y>=aed_lengths 的位置去掉），  
        #     但不再額外篩掉 dur >= D，因為我們已經 clamp 過了
        final_mask = flat_sup  # [B*T_y]

        logits_sel   = flat_logits[final_mask]        # [N, D]
        targets_sel  = clamped_targets[final_mask]    # [N]

        # Debug：印出前幾筆 clamped 之後的 targets
        # print("first few clamped targets_sel:", targets_sel[: min(ctc_lengths_filt[0], targets_sel.numel())])

        # 12) 計算 duration loss
        loss_dur = self.dur_loss_fn(logits_sel, targets_sel)  # 平均 loss

        return loss_dur

    def left_to_right_padding(self, hidden_states, hidden_states_lens):
        """
        将模型输出的 left-padded hidden_states (B, L, D)
        转成 right-padded 形式：有效数据靠前，后面补 0。
        hidden_states_lens: 长度为 B 的 tensor，给出每条序列的真实长度。
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device

        # 1) 計算padding長度
        pad_len = (L - hidden_states_lens).unsqueeze(1)  # [B, 1]

        # 2) 計算索引
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        indices = arange + pad_len                                  # [B, L], dtype long

        # 3) 限制索引範圍
        indices = indices.clamp(min=0, max=L-1)

        # 4) gather 重排
        #    把原 left-padded 的 hidden_states 按 indices 在 dim=1 上重排
        right_padded = torch.gather(
            hidden_states,
            dim=1,
            index=indices.unsqueeze(-1).expand(B, L, D)
        )
        return right_padded
    
    def prepare_mas_input(self,
                          log_probs:      torch.Tensor,   # [B, T_x', V], 已做 log_softmax
                          labels:         torch.Tensor,   # [B, T_y]
                          input_lengths:  torch.Tensor,   # [B], 篩後 CTC frame 數
                          label_lengths:  torch.Tensor,   # [B], AED 真正 token 長度
                          padding_id:     int = 0):
        """
        Args:
            log_probs:      [B, T_x', V]， 已做 log_softmax 的 log-probabilities (GPU)
            labels:         [B, T_y]， 整數 labels 序列 (GPU)
            input_lengths:  [B]， 每個 sample 篩後 CTC frame 長度 (GPU)
            label_lengths:  [B]， 每個 sample AED token 長度 (GPU)
            padding_id:     int， padding token id（CTC blank）
        Returns:
            value: [B, T_y, T_x']， 每個 (token y, frame x') 的 log‐prob
            mask:  [B, T_y, T_x']， 0/1 整數 mask，表示 (y,x') 能不能對齊
        """
        B, T_x_p, V = log_probs.size()   # T_x' 是篩後的 CTC frame 長度，V 是詞彙表大小
        _, T_y      = labels.size()
        device      = log_probs.device

        # (1) 如果 input_lengths, label_lengths 是 list，轉成 GPU Tensor
        if isinstance(input_lengths, list):
            input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        if isinstance(label_lengths, list):
            label_lengths = torch.tensor(label_lengths, dtype=torch.long, device=device)

        # (2) 建立 src_mask、tgt_mask
        #     src_mask[b, x'] = True iff x' < input_lengths[b]
        src_mask = (torch.arange(T_x_p, device=device)[None, :] < input_lengths[:, None])  # [B, T_x']
        #     tgt_mask[b, y]   = True iff y < label_lengths[b]
        tgt_mask = (torch.arange(T_y,  device=device)[None, :] < label_lengths[:, None])  # [B, T_y]
        #     pad_mask[b, y]   = True iff labels[b,y] != padding_id
        pad_mask = (labels != padding_id)                                                   # [B, T_y]
        #     最終有效的 tgt_mask：在長度內且不是 pad
        tgt_mask = tgt_mask & pad_mask                                                       # [B, T_y]

        # (3) gather 出每個 (b, y, x') 的 log‐prob
        #     labels_expanded: [B, T_y, T_x']
        labels_expanded = labels[:, :, None].expand(B, T_y, T_x_p)
        #     logp_expanded:   [B, T_y, T_x', V]
        logp_expanded = log_probs[:, None, :, :].expand(B, T_y, T_x_p, V)
        #     token_logps[b, y, x'] = logp_ctc[b, x', labels[b,y]]
        token_logps = torch.gather(
            logp_expanded,
            dim=3,
            index=labels_expanded[:, :, :, None]
        ).squeeze(3)  # → [B, T_y, T_x']

        # (4) 建立 attn_mask: 只在 src_mask[b,x']=True 且 tgt_mask[b,y]=True 時允許對齊
        #     attn_mask: [B, T_y, T_x']
        attn_mask = (src_mask[:, None, :] & tgt_mask[:, :, None]).to(torch.int32)

        # (5) **保持 [B, T_y, T_x']** 給 maximum_path，切勿轉置
        value = token_logps              # [B, T_y, T_x']
        mask  = attn_mask                # [B, T_y, T_x']

        return value, mask

    def numtoword(self, beam_results, blank_label=0, collapse_repeated=True): # out_lens, labels, label_lengths
        arg_maxes = beam_results

        decodes = []
        for i, args in enumerate(arg_maxes):
            decode = []
        
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j-1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.tokenizer.int_to_text(decode))
        return decodes #, targets

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def print_predict(self, epoch, ar_logits, ctc_logits, ctc_lengths, labels, label_lengths):
        """印出驗證結果（根據 validate 與 test 的流程設計）
        
        參數說明：
        ar_logits : [batch, tgt_seq_len, vocab_size]
        ctc_logits : [src_seq_len, batch, vocab_size]
        ctc_lengths: 若非 None，表示使用 beam search 解碼（一般為 tensor 或 list）；若為 None，則使用 argmax 進行貪婪解碼
        labels     : ground-truth 序列，形狀 [batch, seq_len]
        label_lengths: 每個樣本的 ground-truth 長度（list 或 tensor）
        """

        # --- AR 輸出處理 ---
        # 取 argmax：結果 shape [batch, tgt_seq_len]
        ar_preds = ar_logits.argmax(dim=-1)
        ar_preds = ar_preds.cpu().tolist()
        # print("ar preds:", ar_preds[0])
        
        # --- CTC 輸出處理 ---
        # 計算 log softmax（假設 decoder 需要 log 機率）
        ctc_logits = torch.log_softmax(ctc_logits, dim=-1)

        # 呼叫 beam 解碼器。請根據你使用的解碼器 API 調整參數。
        beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(ctc_logits, ctc_lengths)
        # beam_results shape 假設為 [batch, beam_size, max_decoded_len]；out_lens 為 [batch, beam_size]
        beam_results = beam_results.tolist()
        # print("beam results:", beam_results)

        batch_size = ctc_logits.size(0)
        ctc_preds = []
        ctc_preds = [beam_results[i][0][:out_lens[i][0]] for i in range(batch_size)]
        # print("ctc preds:", ctc_preds[0])

        # --- 轉換為文字 ---
        # AED 預測：過濾掉 padding token（self.pad_id及 self.eos_id）
        ar_texts = []
        for seq in ar_preds:
            filtered_seq = []
            for token in seq:
                if token == self.eos_id:
                    break
                filtered_seq.append(token)
            ar_texts.append(self.tokenizer.int_to_text(filtered_seq))
        
        # CTC 預測：過濾掉 blank token（self.blank_id）
        ctc_texts = []
        for seq in ctc_preds:
            filtered_seq = [token for token in seq if token != self.blank_id]
            ctc_texts.append(self.tokenizer.int_to_text(filtered_seq))
        
        # --- 處理 ground-truth ---
        gt_texts = []
        gt_cpu = labels.cpu().tolist()
        # 假定 label_lengths 是一個列表或 tensor（如果是 tensor，請轉換為 list）
        if isinstance(label_lengths, torch.Tensor):
            label_lengths = label_lengths.cpu().tolist()
        for seq, l in zip(gt_cpu, label_lengths):
            filtered_seq = [token for token in seq[:l] if token != self.pad_id]
            gt_texts.append(self.tokenizer.int_to_text(filtered_seq))
        
        # --- 印出結果 ---
        batch_size = len(gt_texts)

        # 把結果存到檔案
        with open(self.preds_path, "a") as f:
            f.write(f"Epoch {epoch} Predictions:\n")
            f.write("-" * 50 + "\n")
            for i in range(min(3, batch_size)):
                f.write(f"Sample {i + 1}:\n")
                f.write(f"Label: {gt_texts[i]}\n")
                f.write(f"AR: {ar_texts[i]}\n")
                f.write(f"CTC: {ctc_texts[i]}\n")
                f.write("-" * 50 + "\n")
                
                # print(f"Sample {i + 1}:")
                # print(f"Label: {gt_texts[i]}")
                # print(f"AED: {aed_texts[i]}")
                # print(f"CTC: {ctc_texts[i]}")

    def fit(self, epochs, model_path=None, log_path=None, preds_path=None, scaler=None):
        self.scaler = scaler
        model_path = "./weights/" + model_path
        log_path = "./logs/" + log_path
        self.preds_path = "./predictions/" + preds_path

        if log_path and os.path.exists(log_path):
            start_epoch = self.read_log(log_path) + 1

            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            start_epoch = 1

        self.dur_loss_weight = 0.0
        
        """執行多個 epoch 的訓練與驗證"""
        for epoch in range(start_epoch, epochs+1):
            if epoch > epochs / 10:
                self.dur_loss_weight = self.duration_loss_weight

            torch.cuda.empty_cache()
            # 訓練與驗證
            train_loss, train_ctc_loss, train_ce_loss, train_dur_loss = self.train(epoch)
            torch.cuda.empty_cache()
            valid_loss, valid_ctc_loss, valid_ce_loss, valid_dur_loss = self.validate(epoch)
            torch.cuda.empty_cache()

            print(f"Training Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f}, CE: {train_ce_loss:.4f}, Dur: {train_dur_loss:.4f})")
            print(f"Validation Loss: {valid_loss:.4f} (CTC: {valid_ctc_loss:.4f}, CE: {valid_ce_loss:.4f}, Dur: {valid_dur_loss:.4f})")
            
            # 計算 WER
            if epoch > 120:
                ctc_wer, aed_wer = self.test(self.valid_loader)
            else:
                ctc_wer, aed_wer = 100, 100

            # 儲存最佳模型
            if epoch == 1 or (epoch <= 120 and math.isfinite(valid_loss) and valid_loss < min(self.valid_loss_history)):
                self.save_model(model_path)
                print(f"Save epoch {epoch} model as {model_path} with Loss: {valid_loss:.4f}")
            elif aed_wer < min(self.valid_aed_wer_history):
                self.save_model(model_path)
                print(f"Save epoch {epoch} model as {model_path} with WER: {aed_wer:.2f}")

            print("-"*40)

            self.train_loss_history.append(train_loss)
            self.train_ctc_loss_history.append(train_ctc_loss)
            self.train_ce_loss_history.append(train_ce_loss)
            self.train_dur_loss_history.append(train_dur_loss)
            self.valid_loss_history.append(valid_loss)
            self.valid_ctc_loss_history.append(valid_ctc_loss)
            self.valid_ce_loss_history.append(valid_ce_loss)
            self.valid_dur_loss_history.append(valid_dur_loss)

            self.valid_ctc_wer_history.append(ctc_wer)
            self.valid_aed_wer_history.append(aed_wer)

            # 儲存訓練過程的 loss 與 WER 紀錄
            self.save_log(epoch, log_path)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True))