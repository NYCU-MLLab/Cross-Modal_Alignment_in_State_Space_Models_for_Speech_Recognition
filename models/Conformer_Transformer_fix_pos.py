import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conformer.encoder import ConformerEncoder


# ------------------------------
# 產生 Subsequent Mask（因果遮罩）的函數
# def generate_square_subsequent_mask(sz):
#     # 產生一個 bool 型的上三角矩陣：True 表示該位置要被遮蔽
#     mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
#     mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

#     return mask

# ------------------------------
# 固定的 Sinusoidal Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        """
        d_model: 輸入特徵的維度
        dropout: dropout 機率
        max_len: 句子的最大長度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ------------------------------
# 整體 ASR 模型
class Conformer_Transformer_Model_fix_pos_Model(nn.Module):
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
        super(Conformer_Transformer_Model_fix_pos_Model, self).__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        # -----------------------
        # Encoder 部分
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=d_model,
            num_layers=num_encoder_layers,
            num_attention_heads=nhead,
            feed_forward_expansion_factor=4,
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
        # self.learnable_pos = nn.Parameter(torch.zeros(max_len, d_model))
        # nn.init.normal_(self.learnable_pos, mean=0, std=0.1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=F.silu,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.aed_fc = nn.Linear(d_model, vocab_size)

        # 設定 target padding token 索引，假設為 0
        self.trg_pad_idx = 0
        # device 會在 forward 或 generate 中根據輸入自動設定
        self.device = None

    # 定義 make_pad_mask: 輸入序列 [batch, seq_len] ，回傳布林 mask，True 表示該位置為 pad
    def make_pad_mask(self, seq):
        return (seq == self.trg_pad_idx)

    # 定義 make_no_peak_mask: 產生下三角遮罩，形狀 [seq_len, seq_len]
    # def make_no_peak_mask(self, seq_len):
    #     mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
    #     return mask

    # 產生 Subsequent Mask（因果遮罩）的函數
    def generate_square_subsequent_mask(self, sz):
        # 產生一個 bool 型的上三角矩陣：True 表示該位置要被遮蔽
        mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

        return mask

    def forward(self, src, tgt=None, input_lengths=None, label_lengths=None,
                bos_id=1, eos_id=2,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        src: 輸入特徵, shape [batch, src_seq_len, input_dim]
        tgt: 目標 token 序列, shape [batch, tgt_seq_len] (token indices)
        """
        device = src.device
        self.device = device  # 保存 device 方便後續使用

        # -----------------------
        # Encoder 部分：得到 encoder 輸出與更新後的長度
        hidden_states, hidden_states_lens = self.encoder(src, input_lengths)
        # 轉置成 [time, batch, d_model] 供後續 decoder 使用
        hidden_states = hidden_states.transpose(0, 1)
        # 計算 CTC logits，shape: [time, batch, vocab_size]
        ctc_logits = self.ctc_fc(hidden_states)

        # 使用 teacher forcing (訓練階段)
        # 在 target 序列前加入 <bos>
        # aed_input = torch.cat([torch.full((tgt.shape[0], 1), bos_id, device=device), tgt], dim=1)   # [batch, tgt_seq_len+1]

        # Embedding 與加入 learnable positional embedding
        tgt_emb = self.embedding(tgt)  # [batch, tgt_seq_len+1, d_model]
        # pos_embed = self.learnable_pos[:tgt_emb.size(1), :].unsqueeze(0)  # [1, tgt_seq_len+1, d_model]
        pos_embed = self.pos_enc(tgt_emb)
        tgt_emb = tgt_emb + pos_embed

        # 轉置為 [tgt_seq_len+1, batch, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_seq_len = tgt_emb.size(0)

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
            memory_key_padding_mask=src_key_padding_mask  # 可根據需要傳入 encoder 的 padding mask
        )
        # 取得 AED logits，shape: [tgt_seq_len+1, batch, vocab_size]
        aed_logits = self.aed_fc(decoder_output)
        aed_logits = torch.log_softmax(aed_logits, dim=-1)

        return ctc_logits, aed_logits, hidden_states_lens

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
            hidden_states = hidden_states.transpose(0, 1)  # [time, batch, d_model]
            
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
                # pos_embed = self.learnable_pos[:tgt_emb.size(1), :].unsqueeze(0)  # [1, seq_len, d_model]
                pos_embed = self.pos_enc(tgt_emb)
                tgt_emb = tgt_emb + pos_embed
                # 轉置為 [seq_len, batch, d_model] 供 TransformerDecoder 使用
                tgt_emb = tgt_emb.transpose(0, 1)
                # 產生因果遮罩與 padding mask：  
                tgt_causal_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)  # [seq_len, seq_len]
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
                last_step_logits = step_logits[-1, :, :]    # [batch, vocab_size]
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
            padded_sequences = [seq + [self.trg_pad_idx] * (max_len_out - len(seq)) for seq in final_sequences]
            final_tensor = torch.tensor(padded_sequences, device=device, dtype=torch.long)
            
            return final_tensor, ctc_logits, hidden_states_lens
    
