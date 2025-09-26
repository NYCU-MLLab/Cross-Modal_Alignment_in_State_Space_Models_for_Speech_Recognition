import torch
import torch.nn as nn
import torch.nn.functional as F

class EOSEnhancedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=3, label_smoothing=0.0, eos_id=2, eos_weight=2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.eos_id = eos_id
        self.eos_weight = eos_weight

    def forward(self, logits, targets):
        # 1) 原版 cross‐entropy（每個位置一個 loss）
        loss = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )  # shape 與 targets 相同

        weights = torch.ones_like(targets, dtype=loss.dtype, device=loss.device)
        weights = weights.masked_fill(targets == self.ignore_index, 0.0)
        weights = weights.masked_fill(targets == self.eos_id, self.eos_weight)

        total_weight = weights.sum()
        if total_weight == 0:
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        
        eps = 1e-6
        weighted_loss = (loss * weights).sum() / (total_weight + eps)
        return weighted_loss
    
class Focal_loss(nn.Module):
    def __init__(
        self,
        alpha=None,           # 如果傳入 list，也會在這裡轉成 tensor
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        alpha: None 或者 一個長度為 C 的 Python list / torch.Tensor
               代表每個類別的權重係數；如果是 list，會自動轉成 tensor
        gamma: 聚焦係數 (float)
        reduction: 'mean', 'sum' 或 'none'
        label_smoothing: label smoothing 參數
        """
        super(Focal_loss, self).__init__()
        self.reduction      = reduction.lower()
        self.gamma          = gamma

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                alpha = alpha.float()
            else:
                raise ValueError("alpha 必須是 None、長度為 C 的 list 或 torch.Tensor")

            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C] 未經 softmax 的輸出
        targets: [N]   真實標籤，整數 in [0..C-1]
        """

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction='none',
        )  # 形狀 [N]

        pt = torch.exp(-ce_loss).clamp(min=1e-6)

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)  # [C]
            # alpha_t: 針對每個樣本取出對應的類別權重 → 形狀 [N]
            alpha_t = alpha.gather(dim=0, index=targets)
            loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            loss = ((1 - pt) ** self.gamma) * ce_loss  # [N]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss