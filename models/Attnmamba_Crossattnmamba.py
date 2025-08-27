import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from models.conformer.convolution import Conv2dSubampling
from models.mamba.mamba_ssm.modules.biattnmamba2_simple import BiAttnMamba2Simple
from models.mamba.mamba_ssm.modules.crossattnmamba2_simple import CrossAttnMamba2Simple

class mlp(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()  # 使用 GELU 激活函數
        # self.act = nn.ReLU()  # 使用 ReLU 激活函數

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class MambaEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads,feed_forward_dim, dropout, device):
        super(MambaEncoderLayer, self).__init__()
        self.mamba2 = BiAttnMamba2Simple(
            d_model=d_model,
            d_state=64,
            window_size=4,
            expand=2,
            headdim=d_model//num_heads,
            ngroups=1,
            learnable_init_states=False,
            causal=False,
            activation="swish",
            device=device
        )

        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)  # 交叉注意力前的正規化
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)  # 前饋網絡前的正規化
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = mlp(d_model, feed_forward_dim, dropout)

    def forward(self, x, mask, input_idxs=None):
        x = x * mask.unsqueeze(-1)  # 將填充部分設置為0
        x, o1, o2, o3 = self.mamba2(u=self.norm1(x), mask=mask)
        x = x + self.dropout1(x)
        x = x + self.dropout2(self.mlp(self.norm2(x)))

        return x, o1, o2, o3

class MambaEncoder(nn.Module):
    """
    MambaEncoder encoder first processes the input with a convolution subsampling layer and then
    with a number of Mamba2 blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            input_dim: int = 80,
            d_model: int = 512,
            num_layers: int = 4,
            num_heads: int = 4,
            feed_forward_dim: int = 2048,
            dropout_p: float = 0.1,
    ):
        super(MambaEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_subsample = Conv2dSubampling(in_channels=1, freq_bins=80)
        flatten_dim = 32 * (input_dim // 4)  # 80 → 20；32×20＝640
        self.input_projection = nn.Sequential(
            nn.Linear(flatten_dim, d_model),
            nn.Dropout(p=dropout_p),
        )

        self.layers = nn.ModuleList([
            MambaEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout_p,
                device=self.device
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def lengths_to_leftpad_mask(self, lengths, seq_len):
        """
        根據「左填充」規則，把每條序列的有效長度 → Bool mask  
        True = 有效位置，False = Padding 位置
        """
        idx = torch.arange(seq_len, device=self.device)          # [seq_len]
        return idx.unsqueeze(0) >= (seq_len - lengths).unsqueeze(1)  # [B, seq_len]

    def forward(self, inputs, input_lengths):
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        mask = self.lengths_to_leftpad_mask(output_lengths, outputs.size(1))
        output_list = {"outputs": [], "o1": [], "o2": [], "o3": []}

        for layer in self.layers:
            outputs, o1, o2, o3 = layer(outputs, mask)
            # outputs = cp.checkpoint(layer, outputs, mask, use_reentrant=False)

            output_list["outputs"].append(outputs)
            output_list["o1"].append(o1)
            output_list["o2"].append(o2) 
            output_list["o3"].append(o3)

        outputs = self.norm(outputs)

        return outputs, output_lengths, mask, output_list

class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feed_forward_dim, dropout, device):
        super(MambaDecoderLayer, self).__init__()

        # 交叉注意力層 - 處理目標序列與編碼器輸出的交互
        self.crossmamba2 = CrossAttnMamba2Simple(
            d_model=d_model,
            d_state=64,
            window_size=4,
            expand=2,
            headdim=d_model//num_heads,
            ngroups=1,
            learnable_init_states=False,
            causal=True,
            activation="swish",
            device=device
        )

        # 層正規化
        # self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)  # 自注意力前的正規化
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)  # 交叉注意力前的正規化
        self.norm3 = nn.RMSNorm(d_model, elementwise_affine=False)  # 前饋網絡前的正規化
        
        # Dropout 層
        # self.dropout1 = nn.Dropout(dropout)  # 自注意力後的 dropout
        self.dropout2 = nn.Dropout(dropout)  # 交叉注意力後的 dropout
        self.dropout3 = nn.Dropout(dropout)  # 前饋網絡後的 dropout
        
        # 前饋網絡
        self.mlp = mlp(d_model, feed_forward_dim, dropout)

    def forward(self, tgt, memory, mem_mask, tgt_mask):
        # 第一步：自注意力子層
        # tgt = tgt * tgt_mask.unsqueeze(-1)  # 將填充部分設置為0
        # tgt1 = self.norm1(tgt)
        # tgt2 = self.unimamba1(tgt1, mask=tgt_mask)
        # tgt2 = self.unimamba1(tgt1)
        # tgt = tgt + self.dropout1(tgt2)
        
        # 第二步：交叉注意力子層
        tgt = tgt * tgt_mask.unsqueeze(-1)  # 將填充部分設置為0
        tgt1 = self.norm2(tgt)

        mem_len = memory.size(1)
        memory = memory * mem_mask.unsqueeze(-1)  # 將填充部分設置為0

        tgt2 = self.crossmamba2(torch.cat((memory, tgt1), dim=1), mem_len, mem_mask=mem_mask, tgt_mask=tgt_mask)[:, -tgt1.shape[1]:]
        tgt = tgt + self.dropout2(tgt2)

        # 第三步：前饋網絡子層
        tgt1 = self.norm3(tgt)
        tgt2 = self.mlp(tgt1)

        tgt = tgt + self.dropout3(tgt2)

        return tgt
    
class MambaDecoder(nn.Module):
    """
    MambaDecoder 處理編碼器的輸出並生成目標序列。
    
    Args:
        input_dim (int, optional): 輸入向量的維度
        d_model (int, optional): 模型的隱藏維度
        num_layers (int, optional): Mamba decoder 層的數量
        num_heads (int, optional): 注意力頭的數量
        feed_forward_dim (int, optional): 前饋網絡的維度
        dropout_p (float, optional): Dropout 的概率
    
    Inputs: tgt, enc
        - **tgt** (batch, time, dim): 包含目標序列的張量
        - **enc** (batch, time, dim): 包含編碼器輸出的張量
    
    Returns: outputs
        - **outputs** (batch, time, dim): 由 Mamba decoder 產生的輸出張量
    """
    def __init__(
            self,
            input_dim: int = 80,
            d_model: int = 512,
            num_layers: int = 4,
            num_heads: int = 4,
            feed_forward_dim: int = 2048,
            dropout_p: float = 0.1,
    ):
        super(MambaDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            MambaDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout_p,
                device=self.device
            )
            for _ in range(num_layers)
        ])

    def count_parameters(self) -> int:
        """ 計算編碼器的參數數量 """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ 更新編碼器的 dropout 概率 """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, tgt, memory, mem_mask, tgt_mask):
        """
        前向傳播處理目標序列和編碼器輸出。
        
        Args:
            tgt (torch.FloatTensor): 目標序列，形狀為 (batch, seq_length, dimension)
            memory (torch.FloatTensor): 編碼器輸出，形狀為 (batch, seq_length, dimension)
        
        Returns:
            torch.FloatTensor: decoder 的輸出，形狀為 (batch, seq_length, dimension)
        """
        # cat tgt 和 enc
        for layer in self.layers:
            tgt = layer(tgt=tgt, memory=memory, mem_mask=mem_mask, tgt_mask=tgt_mask)
            # tgt = cp.checkpoint(layer, tgt=tgt, memory=memory, mem_mask=mem_mask, tgt_mask=tgt_mask, use_reentrant=False)

        tgt = self.norm(tgt)
        
        return tgt

class Attnmamba_Crossattnmamba_Model(nn.Module):
    def __init__(self, input_dim, vocab_size, duration_size=5,
                 d_model=256, nhead=16,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1):
        """
        input_dim: 輸入特徵的維度 (例如聲學特徵)
        vocab_size: 字彙大小（包括 blank）
        d_model: Transformer 的隱藏層維度
        nhead: 多頭注意力頭數
        num_encoder_layers: Encoder 堆疊層數
        num_decoder_layers: Decoder 堆疊層數
        dim_feedforward: 前向全連接層的隱藏層維度
        dropout: dropout 機率
        """
        super(Attnmamba_Crossattnmamba_Model, self).__init__()
        # -----------------------
        # Encoder 部分
        self.encoder = MambaEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=nhead,
            feed_forward_dim=dim_feedforward,
            dropout_p=dropout,
        )
        self.ctc_fc = nn.Linear(d_model, vocab_size)

        # -----------------------
        # Decoder 部分（AED 分支）
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = MambaDecoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=nhead,
            feed_forward_dim=dim_feedforward,
            dropout_p=dropout,
        )
        self.aed_fc = nn.Linear(d_model, vocab_size + duration_size)  # 包括 duration token

        # device 會在 forward 或 generate 中根據輸入自動設定
        self.device = None
        self.vocab_size = vocab_size
        self.duration_size = duration_size  # 用於 duration token 的大小

        self.sos_id = 1
        self.eos_id = 2
        self.pad_id = 3
        self.blank_id = 4

    # 定義 make_pad_mask: 輸入序列 [batch, seq_len] ，回傳布林 mask，True 表示該位置為 pad
    def make_pad_mask(self, seq):
        return (seq == self.pad_id)

    def forward(self, src, tgt=None, input_lengths=None, label_lengths=None, tgt_pad_lengths=None):
        """
        src: 輸入特徵, shape [batch, src_seq_len, input_dim]
        tgt: 目標 token 序列, shape [batch, tgt_seq_len] (token indices)
        """
        self.device = src.device  # 保存 device 方便後續使用

        # -----------------------
        # Encoder 部分：得到 encoder 輸出與更新後的長度
        hidden_states, hidden_states_lens, hidden_states_mask, output_list = self.encoder(src, input_lengths)

        # 計算 CTC logits，shape: [time, batch, vocab_size]
        ctc_logits = self.ctc_fc(hidden_states)

        tgt_mask = self.make_pad_mask(tgt)  # [batch, tgt_seq_len]
        tgt_mask = ~tgt_mask.to(self.device)
        tgt_emb = self.embedding(tgt)  # [batch, tgt_seq_len, d_model]

        decoder_output = self.decoder(
            tgt=tgt_emb,
            memory=hidden_states,
            mem_mask=hidden_states_mask,
            tgt_mask=tgt_mask,
        )

        # 取得 AED logits，shape: [tgt_seq_len+1, batch, vocab_size]
        aed_logits = self.aed_fc(decoder_output)
        
        return ctc_logits, aed_logits, hidden_states_lens, output_list
       
    @torch.no_grad()
    def fast_generate(
        self, src, input_lens,
        beam_size: int = 5,
        ctc_weight: float = 0.3,
        max_len: int = 1024,
    ):
        """
        Beam search with CTC prefix scoring for Attnmamba_Crossattnmamba.

        Args
        ----
        src : FloatTensor [B,T_in,D]
            acoustic feature (left-padded)
        input_lens : LongTensor [B]
            real length before padding
        """

        dev = src.device
        # ---------- Encoder ----------
        enc_out, enc_lens, enc_mask = self.encoder(src, input_lens)              # enc_mask: True=valid
        ctc_logits = self.ctc_fc(enc_out)                                        # [B,T_enc,V]
        logp_ctc = F.log_softmax(ctc_logits, -1)

        B, T_enc, _ = ctc_logits.shape

        # --------- Init beam ----------
        alive_seq  = torch.full((B, 1), self.sos_id, device=dev, dtype=torch.long)
        alive_att  = torch.zeros(B, device=dev)          # AED score
        alive_ctc  = torch.zeros(B, device=dev)          # CTC score
        batch_idx  = torch.arange(B, device=dev)
        finished   = [None] * B                          # save final hyp

        # --------- Main loop ----------
        for _ in range(max_len):
            mem       = enc_out[batch_idx]               # [N_beam,T_enc,d]
            mem_mask  = enc_mask[batch_idx]              # [N_beam,T_enc]  True=valid

            # ---- Decoder (Cross-Mamba) ----
            emb = self.embedding(alive_seq)              # [N_beam,L,d]
            # causal mask : upper-tri = True (=masked)
            L = emb.size(1)
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1)
            tgt_pad = ~(alive_seq != self.pad_id)   # bool pad mask
            tgt_mask = (~tgt_pad)                        # True=valid
            dec_out = self.decoder(
                tgt=emb,
                memory=mem,
                mem_mask=mem_mask,
                tgt_mask=tgt_mask,
            )
            logits = self.aed_fc(dec_out)[:, -1]         # [N_beam,V+D]
            logits = logits[..., :-self.duration_size]                      # [N_beam,V] 

            logits[..., self.pad_id] = -6.5e4       # 禁止 <pad>
            lp_att = F.log_softmax(logits, -1)

            # ---- Top-K expand ----
            top_lp, top_tok = torch.topk(lp_att, beam_size)   # [N,K]
            N, K = top_tok.shape
            new_seq = torch.cat(
                [alive_seq.repeat_interleave(K, 0), top_tok.view(-1, 1)], 1
            )
            new_att = (alive_att.unsqueeze(1) + top_lp).reshape(-1)
            new_bid = batch_idx.repeat_interleave(K)

            # ---- CTC prefix score (1-step approx) ----
            last_tok = new_seq[:, -1]
            has_prev = torch.tensor(new_seq.size(1) > 1, device=new_seq.device)
            prev_tok = torch.where(
                has_prev,
                new_seq[:, -2],
                torch.full_like(last_tok, -1)
            )
            repeat_m = last_tok == prev_tok

            alpha_b  = logp_ctc[new_bid, enc_lens[new_bid] - 1, self.pad_id]
            p_tok    = logp_ctc[new_bid, enc_lens[new_bid] - 1].gather(
                1, last_tok.unsqueeze(1)
            ).squeeze(1)

            add_b  = alpha_b + p_tok
            add_nb = -1e9 + p_tok
            alpha_nb = torch.where(repeat_m, add_b, torch.logaddexp(add_b, add_nb))
            new_ctc = torch.logaddexp(alpha_b, alpha_nb)

            # ---- Joint score & prune ----
            joint = (1 - ctc_weight) * new_att + ctc_weight * new_ctc
            order = joint.argsort(descending=True)

            new_seq, new_att, new_ctc, new_bid = (
                new_seq[order], new_att[order], new_ctc[order], new_bid[order]
            )

            # ---- Re-gather alive ----
            alive_seq, alive_att, alive_ctc, batch_idx = [], [], [], []
            for b in range(B):
                if finished[b] is not None:
                    continue
                sel = (new_bid == b).nonzero(as_tuple=False)[:beam_size].squeeze(1)
                if sel.numel() == 0:
                    continue
                seq_b = new_seq[sel]
                if seq_b[0, -1] == self.eos_id:
                    hyp = seq_b[0, 1:].tolist()
                    if self.eos_id in hyp:
                        hyp = hyp[:hyp.index(self.eos_id)]
                    finished[b] = hyp
                else:
                    alive_seq.append(seq_b)
                    alive_att.append(new_att[sel])
                    alive_ctc.append(new_ctc[sel])
                    batch_idx.append(torch.full((sel.size(0),), b, device=dev))

            if not alive_seq:         # 全 batch 結束
                break
            alive_seq = torch.cat(alive_seq, 0)
            alive_att = torch.cat(alive_att, 0)
            alive_ctc = torch.cat(alive_ctc, 0)
            batch_idx = torch.cat(batch_idx, 0)

        # ---- finalize ----
        hyps = []
        for b in range(B):
            if finished[b] is None:
                cand = alive_seq[batch_idx == b][0][1:].tolist()
                if self.eos_id in cand:
                    cand = cand[:cand.index(self.eos_id)]
                finished[b] = cand
            hyps.append(finished[b])

        return hyps, ctc_logits, enc_lens