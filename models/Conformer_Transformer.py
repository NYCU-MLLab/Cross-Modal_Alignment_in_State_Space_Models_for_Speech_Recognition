import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from models.conformer.encoder import ConformerEncoder

# ------------------------------
# 固定的 Sinusoidal Positional Encoding
def positional_encoding(max_len, d_model):
  pe = np.zeros((max_len, d_model))
  position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
  div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))

  pe[:, 0::2] = np.sin(position * div_term) # 偶數維度
  pe[:, 1::2] = np.cos(position * div_term) # 奇數維度

  return pe

# ------------------------------
# 整體 ASR 模型
class Conformer_Transformer_Model(nn.Module):
    def __init__(self, input_dim, vocab_size,
                 d_model=256, nhead=16,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_len=2000):
        """
        input_dim: 輸入特徵的維度 (例如聲學特徵)
        vocab_size: 字彙大小（包括 blank）
        d_model: Transformer 的隱藏層維度
        nhead: 多頭注意力頭數
        num_encoder_layers: Encoder 堆疊層數
        num_decoder_layers: Decoder 堆疊層數
        dim_feedforward: 前向全連接層的隱藏層維度
        dropout: dropout 機率
        max_len: 最大序列長度 (用於 positional encoding / learnable positional embedding)
        """
        super(Conformer_Transformer_Model, self).__init__()
        # self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        # -----------------------
        # Encoder 部分
        feed_forward_expansion_factor = int(dim_feedforward / d_model)

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=d_model,
            num_layers=num_encoder_layers,
            num_attention_heads=nhead,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=2,
            input_dropout_p=dropout,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
            conv_kernel_size=31,
            half_step_residual=False,
        )
        self.ctc_fc = nn.Linear(d_model, vocab_size)

        # -----------------------
        # Decoder 部分（AED 分支）
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 使用可訓練的位置編碼（learnable positional embedding）
        pe = positional_encoding(max_len, d_model)
        self.register_buffer("pos_enc", torch.from_numpy(pe).float())
        # self.learnable_pos = nn.Parameter(torch.from_numpy(pe).float())
        # nn.init.normal_(self.learnable_pos, mean=0, std=0.1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=F.silu,
            dropout=dropout,
            batch_first=True  # 設定為 True 以符合 [batch, seq_len, feature] 的輸入格式
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # self.decoder_layers = nn.ModuleList([
        #     nn.TransformerDecoderLayer(
        #         d_model=d_model,
        #         nhead=nhead,
        #         dim_feedforward=dim_feedforward,
        #         activation=F.silu,
        #         dropout=dropout,
        #         batch_first=True
        #     )
        #     for _ in range(num_decoder_layers)
        # ])
        self.aed_fc = nn.Linear(d_model, vocab_size)

        # 設定 target padding token 索引，假設為 0
        self.sos_id = 1
        self.eos_id = 2
        self.pad_id = 3
        self.blank_id = 4
        # device 會在 forward 或 generate 中根據輸入自動設定
        self.device = None

    # 定義 make_pad_mask: 輸入序列 [batch, seq_len] ，回傳布林 mask，True 表示該位置為 pad
    def make_pad_mask(self, seq):
        return (seq == self.pad_id)
    
    def make_src_pad_mask(self, lengths, max_len):
        # 建一條遞增向量，broadcast 一次成 2-D，不用 for-loop
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)    # [B, T]  True=pad

    # 產生 Subsequent Mask（因果遮罩）的函數
    def generate_square_subsequent_mask(self, sz):
        # 產生一個 bool 型的上三角矩陣：True 表示該位置要被遮蔽
        mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
        # mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

        return mask

    # def checkpointed_decoder_layer(self, layer, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
    #     def custom_forward(*inputs):
    #         tgt, memory = inputs[0], inputs[1]
    #         return layer(
    #             tgt,
    #             memory,
    #             tgt_mask=tgt_mask,
    #             tgt_key_padding_mask=tgt_key_padding_mask,
    #             memory_key_padding_mask=memory_key_padding_mask,
    #         )
    #     return cp.checkpoint(custom_forward, tgt, memory, use_reentrant=False)

    def forward(self, src, tgt=None, input_lengths=None, label_lengths=None,):
        """
        src: 輸入特徵, shape [batch, src_seq_len, input_dim]
        tgt: 目標 token 序列, shape [batch, tgt_seq_len] (token indices)
        """
        device = src.device
        self.device = device  # 保存 device 方便後續使用

        # -----------------------
        # Encoder 部分：得到 encoder 輸出與更新後的長度
        hidden_states, hidden_states_lens = self.encoder(src, input_lengths)

        # 計算 CTC logits，shape: [batch, time, vocab_size]
        ctc_logits = self.ctc_fc(hidden_states)

        src_pad_mask = self.make_src_pad_mask(hidden_states_lens, hidden_states.size(1)).to(device)  # [batch, T_src]
        # print(f"ctc logits shape:", ctc_logits.shape)  # Debug: 印出 CTC logits 的形狀
        # print(f"hidden_states_lens:", hidden_states_lens[0])  # Debug: 印出 hidden_states_lens 的值
        # print(f"ctc logits:", ctc_logits[0, :, :1])  # Debug: 印出 CTC logits 的前幾個值
        # print(f"src_pad_mask:", src_pad_mask[0, :])  # Debug: 印出 src_pad_mask 的前幾個值

        # 使用 teacher forcing (訓練階段)
        # tgt_emb = self.embedding(tgt)  # [batch, tgt_seq_len+1, d_model]
        # # Embedding 與加入 learnable positional embedding
        # pos_embed = self.learnable_pos[:tgt_emb.size(1), :].unsqueeze(0)  # [1, tgt_seq_len+1, d_model]
        # tgt_emb = tgt_emb + pos_embed
        
        pos_embed = self.pos_enc[:tgt.size(1)].unsqueeze(0)      # [1, L, d_model]
        tgt_emb   = self.embedding(tgt) + pos_embed

        tgt_seq_len = tgt_emb.size(1)

        # 產生因果遮罩與 padding mask：
        # 因果遮罩 shape: [tgt_seq_len, tgt_seq_len]
        tgt_causal_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(device)
        # padding mask: 從 aed_input (shape: [batch, tgt_seq_len+1]) 產生
        tgt_pad_mask = self.make_pad_mask(tgt)  # shape: [batch, tgt_seq_len+1]

        # 傳入 TransformerDecoder：
        decoder_output = self.decoder(
            tgt=tgt_emb,
            memory=hidden_states,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        # decoder_output = tgt_emb
        # for layer in self.decoder_layers:
        #     decoder_output = self.checkpointed_decoder_layer(
        #         layer,
        #         decoder_output,
        #         hidden_states,
        #         tgt_causal_mask,
        #         tgt_pad_mask,
        #         src_pad_mask
        #     )
        # 取得 AED logits，shape: [tgt_seq_len+1, batch, vocab_size]
        aed_logits = self.aed_fc(decoder_output)

        return ctc_logits, aed_logits, hidden_states_lens

    '''
    def generate(self, src, input_lengths, bos_id=1, eos_id=2, beam_size=1):
        """
        Inference 解碼方法
         - src: [batch, src_seq_len, input_dim]
         - input_lengths: 每筆資料的實際 src 長度 (tensor)
         - bos_id, eos_id: 起始與結束 token id
         - beam_size: beam search 的大小；當 beam_size==1 時即為貪婪解碼
         
        回傳：生成的 token 序列，形狀 [batch, max_len_out] (已 padding)
        """
        with torch.no_grad():
            device = src.device
            self.device = device  # 更新 device
            # 取得 encoder 輸出
            hidden_states, hidden_states_lens = self.encoder(src, input_lengths)
            # hidden_states = hidden_states.transpose(0, 1)  # [time, batch, d_model]
            
            # CTC 分支輸出
            ctc_logits = self.ctc_fc(hidden_states)

            # AED 分支輸出
            batch_size = src.size(0)
            max_decoding_steps = 256
            # max_decoding_steps = self.learnable_pos.size(0)
            
            # 貪婪解碼
            # 初始化所有序列，shape: [batch, 1]，每個序列第一個 token 為 bos_id
            greedy_seq = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
            # 用來標記哪些序列已生成 eos（True 表示已結束）
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for t in range(max_decoding_steps):
                # 取得目前 decoder 輸入的 embedding，形狀: [batch, seq_len, d_model]
                tgt_emb = self.embedding(greedy_seq)
                # 加上 learnable positional embedding（依據目前序列長度）
                pos_embed = self.learnable_pos[:tgt_emb.size(1), :].unsqueeze(0)  # [1, seq_len, d_model]
  
                tgt_emb = tgt_emb + pos_embed
                # 轉置為 [seq_len, batch, d_model] 供 TransformerDecoder 使用
                # tgt_emb = tgt_emb.transpose(0, 1)
                # 產生因果遮罩與 padding mask：  
                tgt_causal_mask = self.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)  # [seq_len, seq_len]
                tgt_pad_mask = self.make_pad_mask(greedy_seq).to(device)              # [batch, seq_len]
                
                # 前向傳播得到 decoder 輸出
                decoder_output = self.decoder(
                    tgt=tgt_emb,
                    memory=hidden_states,
                    tgt_mask=tgt_causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=None
                )
                step_logits = self.aed_fc(decoder_output)  # [seq_len, batch, vocab_size]
                last_step_logits = step_logits[:, -1, :]    # [batch, vocab_size]
                # 以貪婪解碼選取下個 token (對整個 batch 同時計算)
                next_tokens = torch.argmax(last_step_logits, dim=-1, keepdim=True)  # [batch, 1]
                
                # 更新 finished mask：若某筆生成了 eos, 則標記為 True
                finished_mask = finished_mask | (next_tokens.squeeze(-1) == eos_id)
                # 對於已完成的樣本，固定其後續輸出為 eos
                next_tokens[finished_mask.unsqueeze(-1)] = eos_id
                
                # 將下個 token 加入所有序列
                greedy_seq = torch.cat([greedy_seq, next_tokens], dim=1)
                # 若所有樣本皆生成 eos，則提前停止
                if finished_mask.all():
                    break
            
            final_sequences = greedy_seq.cpu().tolist()
            
            # Padding 序列到相同長度
            max_len_out = max(len(seq) for seq in final_sequences)
            padded_sequences = [seq + [self.pad_id] * (max_len_out - len(seq)) for seq in final_sequences]
            final_tensor = torch.tensor(padded_sequences, device=device, dtype=torch.long)
            
            return final_tensor, ctc_logits, hidden_states_lens
    '''

    @torch.no_grad()
    def fast_generate(
        self, src, input_lens,
        beam_size=5, ctc_weight=.3,
        max_len=256):

        dev = src.device
        enc_out, enc_lens = self.encoder(src, input_lens)      # [B,T,d]
        ctc_logits = self.ctc_fc(enc_out)                      # [B,T,V]
        logp_ctc   = F.log_softmax(ctc_logits, -1)

        B, T_enc, _ = ctc_logits.shape

        src_pad_mask = self.make_src_pad_mask(enc_lens, T_enc)   # [B,T]  GPU

        # -------- 初始 beam --------
        alive_seq = torch.full((B, 1), self.sos_id, device=dev, dtype=torch.long)
        alive_att = torch.zeros(B, device=dev)
        alive_ctc = torch.zeros(B, device=dev)                 # ← 保留
        batch_idx = torch.arange(B, device=dev)
        finished  = [None] * B

        # -------- 主迴圈 --------
        for _ in range(max_len):
            mem = enc_out[batch_idx]          # [N_beam,T_enc,d]
            mem_mask = src_pad_mask[batch_idx]     # [N_beam,T_enc] 0 copy

            # ---- Decoder ----
            emb  = self.embedding(alive_seq) + self.pos_enc[:alive_seq.size(1)].to(dev)
            tgt_mask = self.generate_square_subsequent_mask(alive_seq.size(1)).to(dev)

            dec_out = self.decoder(
                emb, mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,         # 無 <pad>，省略
                memory_key_padding_mask=mem_mask
            )

            lp_att  = F.log_softmax(self.aed_fc(dec_out)[:, -1], -1)  # [N,V]

            out = self.aed_fc(dec_out)[:, -1]
            out[..., self.blank_id] = -6.5e4          # 4 = <blank>
            lp_att = F.log_softmax(out, -1)

            # ------ Top-K 擴展 -------
            top_lp, top_tok = torch.topk(lp_att, beam_size)           # [N,K]
            N, K   = top_tok.shape
            new_seq = torch.cat([alive_seq.repeat_interleave(K, 0),
                                top_tok.view(-1, 1)], 1)             # [N*K,L+1]
            new_att = (alive_att.unsqueeze(1) + top_lp).reshape(-1)
            new_bid = batch_idx.repeat_interleave(K)

            # ------ CTC 前綴計分 ------
            last_tok = new_seq[:, -1]

            if new_seq.size(1) > 1:           # L>1 才有 prev_tok
                prev_tok  = new_seq[:, -2]
                repeat_m  = last_tok == prev_tok
            else:                             # L==1 只含 <bos>
                prev_tok  = torch.full_like(last_tok, -1)
                repeat_m  = torch.zeros_like(last_tok, dtype=torch.bool)

            alpha_b  = torch.zeros_like(new_att)
            alpha_nb = torch.full_like(new_att, -1e9)

            # 只取最後一個有效 encoder 幀 (近似)
            t_idx = enc_lens[new_bid] - 1                     # [N*K]
            p_t   = logp_ctc[new_bid, t_idx]                  # [N*K,V]

            alpha_b += p_t[:, self.blank_id]
            p_tok    = p_t.gather(1, last_tok.unsqueeze(1)).squeeze(1)
            add_b, add_nb = alpha_b + p_tok, alpha_nb + p_tok
            alpha_nb = torch.where(repeat_m, add_b,
                                torch.logaddexp(add_b, add_nb))
            new_ctc  = torch.logaddexp(alpha_b, alpha_nb)

            # ------ Beam 剪枝 ----------
            joint  = (1 - ctc_weight) * new_att + ctc_weight * new_ctc
            order  = joint.argsort(descending=True)
            new_seq, new_att, new_ctc, new_bid = \
                new_seq[order], new_att[order], new_ctc[order], new_bid[order]

            # ------ 重建活路徑 ----------
            alive_seq, alive_att, alive_ctc, batch_idx = [], [], [], []
            for b in range(B):
                if finished[b] is not None:
                    continue
                sel = (new_bid == b).nonzero(as_tuple=False)[:beam_size].squeeze(1)
                if sel.numel() == 0:
                    continue

                seq_b = new_seq[sel]
                if seq_b[0, -1] == self.eos_id:        # 已完成
                    hyp = seq_b[0, 1:].tolist()
                    if self.eos_id in hyp:
                        hyp = hyp[:hyp.index(self.eos_id)]
                    finished[b] = hyp
                else:                             # 還沒完成，留下來繼續擴展
                    alive_seq.append(seq_b)
                    alive_att.append(new_att[sel])
                    alive_ctc.append(new_ctc[sel])
                    batch_idx.append(torch.full((sel.size(0),), b, device=dev))

            if not alive_seq:                     # 全 batch 均已結束
                break

            alive_seq = torch.cat(alive_seq, 0)
            alive_att = torch.cat(alive_att, 0)
            alive_ctc = torch.cat(alive_ctc, 0)
            batch_idx = torch.cat(batch_idx, 0)

        # ----- 收尾：把仍未結束者補完 -----
        hyps = []
        for b in range(B):
            if finished[b] is None:
                cand = alive_seq[batch_idx == b][0][1:].tolist()
                if self.eos_id in cand:
                    cand = cand[:cand.index(self.eos_id)]
                finished[b] = cand
            hyps.append(finished[b])

        return hyps, ctc_logits, enc_lens
