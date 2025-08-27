import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from models.conformer.encoder import ConformerEncoder

# ------------------------------
# 整體 ASR 模型
class Conformer_CTC_Model(nn.Module):
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
        super(Conformer_CTC_Model, self).__init__()
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

        # device 會在 forward 或 generate 中根據輸入自動設定
        self.device = None

    def forward(self, src, input_lengths=None):
        """
        src: 輸入特徵, shape [batch, src_seq_len, input_dim]
        tgt: 目標 token 序列, shape [batch, tgt_seq_len] (token indices)
        """
        device = src.device
        self.device = device  # 保存 device 方便後續使用

        # -----------------------
        # Encoder 部分：得到 encoder 輸出與更新後的長度
        # hidden_states, hidden_states_lens = self.encoder(src, input_lengths)

        hidden_states, hidden_states_lens = cp.checkpoint(
            self.encoder, src, input_lengths, use_reentrant=False
        )

        # 計算 CTC logits，shape: [batch, time, vocab_size]
        ctc_logits = self.ctc_fc(hidden_states)

        return ctc_logits, hidden_states_lens
