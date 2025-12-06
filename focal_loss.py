import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss - citation:
# Lin et al., 2017
# Focal Loss for Dense Object Detection
# https://arxiv.org/abs/1708.02002

# Binary crossentropy loss:
#      BCE(p, y) = - [ y * log(p) + (1 - y) * log(1 - p) ]
#             where y in {0,1} is the target class and
#             p in [0,1] is the model's estimated probability for class 1.
#   It can be written as:
#      BCE(p_t) = - log(p_t)
#   where
#      p_t = { p  , if y = 1
#              1-p, if y = 0 }
# Focal loss:
#      FL(p_t) = - alpha_t * (1 - p_t)^{gamma} * log(p_t)
#   where
#      p_t = { p  , if y = 1
#              1-p, if y = 0 }
#   and
#      alpha_t = { alpha, if y = 1
#                  1-alpha, if y = 0 }
#
#   alpha in [0,1] is the weighting factor for class 1,
#   gamma >= 0 is the focusing parameter.
# 
# To remain numerically stable I used PyTorch's binary_cross_entropy_with_logits loss function.
#
# The implemented focal loss therefore is:
#   FL(x, y) = alpha_t * (1 - p_t)^{\gamma} * BCE_with_logits(x, y)

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (PyTorch).

    This implementation accepts logits (raw model outputs). It is numerically
    stable because it uses binary_cross_entropy_with_logits internally.

    Args:
        gamma (float): Focusing parameter, gamma=0 reduces to BCEWithLogitsLoss.
        alpha (float or None): Weighting factor for the positive class (class=1).
            If None, no class weighting is applied. If float, alpha should be in [0,1]
            and will weight positives by alpha and negatives by (1-alpha).
        reduction (str): 'mean', 'sum' or 'none'. Default 'mean'.
        pos_weight (Tensor or None): A weight of positive examples. Passed to
            binary_cross_entropy_with_logits if provided (same semantics as PyTorch).
    Returns:
        Scalar loss (if reduction != 'none') or elementwise loss tensor.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean', pos_weight=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.pos_weight = pos_weight

        if alpha is not None:
            # store alpha as tensor lazily moved to device in forward
            if not isinstance(alpha, (float, int, torch.Tensor)):
                raise ValueError("alpha must be float, int, torch.Tensor or None")
            self.alpha = torch.tensor(float(alpha)) if not isinstance(alpha, torch.Tensor) else alpha
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        logits: any shape, raw model outputs (not sigmoid-ed)
        targets: same shape as logits (or broadcastable), 0/1 values
        """
        if not (logits.shape == targets.shape or logits.shape == targets.shape + (1,)):
            # allow (N,) vs (N,1) shapes as common in binary setups
            try:
                targets = targets.view_as(logits)
            except Exception:
                pass

        targets = targets.type_as(logits)

        # BCE with logits (elementwise)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)

        # Probabilities for positive class
        prob = torch.sigmoid(logits)
        # p_t is p if y==1 else 1-p
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # Focusing factor
        mod_factor = (1.0 - p_t) ** self.gamma

        loss = mod_factor * bce_loss

        # Alpha balancing
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device).type_as(logits)
            alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
            loss = alpha_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
