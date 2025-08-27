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
        x = x + self.dropout1(self.mamba2(u=self.norm1(x), mask=mask))
        x = x + self.dropout2(self.mlp(self.norm2(x)))

        return x

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

        for layer in self.layers:
            outputs = layer(outputs, mask)
            # outputs = cp.checkpoint(layer, outputs, mask, use_reentrant=False)

        outputs = self.norm(outputs)

        return outputs, output_lengths, mask

class Attnmamba_CTC_Model(nn.Module):
    def __init__(self, input_dim, vocab_size,
                 d_model=256, nhead=16,
                 num_encoder_layers=3,
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
        super(Attnmamba_CTC_Model, self).__init__()
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

        # device 會在 forward 或 generate 中根據輸入自動設定
        self.device = None

    def forward(self, src, input_lengths=None):
        """
        src: 輸入特徵, shape [batch, src_seq_len, input_dim]
        tgt: 目標 token 序列, shape [batch, tgt_seq_len] (token indices)
        """
        self.device = src.device  # 保存 device 方便後續使用

        # -----------------------
        # Encoder 部分：得到 encoder 輸出與更新後的長度
        hidden_states, hidden_states_lens, _ = self.encoder(src, input_lengths)

        # 計算 CTC logits，shape: [time, batch, vocab_size]
        ctc_logits = self.ctc_fc(hidden_states)
        
        return ctc_logits, hidden_states_lens
