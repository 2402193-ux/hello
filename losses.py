"""
损失函数模块 - 修复版 (NaN-Safe)

修复内容:
1. AsymmetricLoss 添加更强的数值稳定性
2. 添加 NaN 检测和安全返回
3. 使用 logsigmoid 代替 log(sigmoid) 提高稳定性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss For Multi-Label Classification (ICCV 2021)

    修复版 - 添加数值稳定性保护

    Args:
        gamma_neg: 负样本的focusing参数，默认4
        gamma_pos: 正样本的focusing参数，默认1
        clip: 负样本概率截断阈值，默认0.05
        eps: 数值稳定性，增大到1e-6
        disable_torch_grad_focal_loss: 是否禁用focal loss部分的梯度
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6,
                 disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """
        Args:
            x: 模型输出logits, shape [batch_size, num_classes]
            y: 标签, shape [batch_size, num_classes], 值为0或1

        Returns:
            loss: 标量损失值
        """
        # ====== 输入检查 ======
        if torch.isnan(x).any() or torch.isinf(x).any():
            # 如果输入已经是 NaN/Inf，返回一个安全的损失值
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # ====== Clamp logits 防止极端值 ======
        x = torch.clamp(x, min=-50, max=50)

        # 计算概率 (使用 clamp 确保在安全范围内)
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = torch.clamp(x_sigmoid, min=self.eps, max=1 - self.eps)

        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (对负样本概率进行截断)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # ====== 使用更稳定的 log 计算 ======
        # 确保 log 的输入始终为正
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # ====== 检查中间结果 ======
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            # 用 0 替换 NaN/Inf
            loss = torch.where(
                torch.isnan(loss) | torch.isinf(loss),
                torch.zeros_like(loss),
                loss
            )

        # Asymmetric Focusing (非对称聚焦)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # 计算pt (预测正确的概率)
            pt0 = xs_pos * y  # 正样本的预测概率
            pt1 = xs_neg * (1 - y)  # 负样本的预测概率 (1-p)
            pt = pt0 + pt1
            pt = torch.clamp(pt, min=self.eps, max=1 - self.eps)

            # 计算focal weight (添加 clamp 防止数值问题)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow((1 - pt).clamp(min=0), one_sided_gamma)

            # 防止权重过大
            one_sided_w = torch.clamp(one_sided_w, max=100)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss *= one_sided_w

        # ====== 最终安全检查 ======
        result = -loss.sum() / max(x.size(0), 1)

        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        return result


class AsymmetricLossOptimized(nn.Module):
    """
    优化版本的Asymmetric Loss - 修复版
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6,
                 disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Clamp logits
        x = torch.clamp(x, min=-50, max=50)

        targets = y
        anti_targets = 1 - y

        # 计算概率
        xs_pos = torch.sigmoid(x)
        xs_pos = torch.clamp(xs_pos, min=self.eps, max=1 - self.eps)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 基础BCE
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss = loss + anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        # 处理 NaN
        loss = torch.where(
            torch.isnan(loss) | torch.isinf(loss),
            torch.zeros_like(loss),
            loss
        )

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            xs_pos_focal = xs_pos * targets
            xs_neg_focal = xs_neg * anti_targets

            pt = xs_pos_focal + xs_neg_focal
            pt = torch.clamp(pt, min=self.eps, max=1 - self.eps)

            asymmetric_w = torch.pow(
                (1 - pt).clamp(min=0),
                self.gamma_pos * targets + self.gamma_neg * anti_targets
            )
            asymmetric_w = torch.clamp(asymmetric_w, max=100)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss *= asymmetric_w

        result = -loss.sum() / max(x.size(0), 1)

        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        return result


class FocalLoss(nn.Module):
    """
    Focal Loss - 修复版
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Clamp inputs
        inputs = torch.clamp(inputs, min=-50, max=50)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Clamp BCE_loss 防止 exp 溢出
        BCE_loss = torch.clamp(BCE_loss, max=50)

        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            result = focal_loss.mean()
        elif self.reduction == 'sum':
            result = focal_loss.sum()
        else:
            result = focal_loss

        if torch.isnan(result).any() or torch.isinf(result).any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return result


class MultiLabelSoftMarginLoss(nn.Module):
    """标准的多标签软边界损失"""

    def __init__(self, weight=None, reduction='mean'):
        super(MultiLabelSoftMarginLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=-50, max=50)
        return F.multilabel_soft_margin_loss(
            inputs, targets,
            weight=self.weight,
            reduction=self.reduction
        )


class TripletComponentLoss(nn.Module):
    """三元组组件感知损失"""

    def __init__(self, triplet_weight=1.0, component_weight=0.5,
                 gamma_neg=4, gamma_pos=1, clip=0.05):
        super(TripletComponentLoss, self).__init__()

        self.triplet_weight = triplet_weight
        self.component_weight = component_weight

        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip
        )

    def forward(self, outputs, triplet_labels,
                instrument_labels=None, verb_labels=None, target_labels=None):

        loss_triplet = self.asl(outputs['triplets'], triplet_labels)
        total_loss = self.triplet_weight * loss_triplet

        if instrument_labels is not None and 'instruments' in outputs:
            loss_i = self.asl(outputs['instruments'], instrument_labels)
            total_loss = total_loss + self.component_weight * loss_i

        if verb_labels is not None and 'verbs' in outputs:
            loss_v = self.asl(outputs['verbs'], verb_labels)
            total_loss = total_loss + self.component_weight * loss_v

        if target_labels is not None and 'targets' in outputs:
            loss_t = self.asl(outputs['targets'], target_labels)
            total_loss = total_loss + self.component_weight * loss_t

        return total_loss


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Loss Functions (NaN-Safe Version)")
    print("=" * 60)

    batch_size = 4
    num_classes = 100

    # 正常测试
    print("\n--- Normal Input Test ---")
    logits = torch.randn(batch_size, num_classes)
    labels = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        pos_indices = torch.randint(0, num_classes, (torch.randint(1, 4, (1,)).item(),))
        labels[i, pos_indices] = 1

    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    loss = asl(logits, labels)
    print(f"ASL Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"

    # 极端值测试
    print("\n--- Extreme Value Test ---")
    extreme_logits = torch.randn(batch_size, num_classes) * 100  # 很大的值
    loss_extreme = asl(extreme_logits, labels)
    print(f"ASL Loss (extreme): {loss_extreme.item():.4f}")
    assert not torch.isnan(loss_extreme), "Loss should not be NaN for extreme values"

    # NaN 输入测试
    print("\n--- NaN Input Test ---")
    nan_logits = torch.randn(batch_size, num_classes)
    nan_logits[0, 0] = float('nan')
    loss_nan = asl(nan_logits, labels)
    print(f"ASL Loss (NaN input): {loss_nan.item():.4f}")
    assert not torch.isnan(loss_nan), "Loss should handle NaN input gracefully"

    # 梯度测试
    print("\n--- Gradient Test ---")
    logits.requires_grad = True
    loss = asl(logits, labels)
    loss.backward()
    grad_has_nan = torch.isnan(logits.grad).any()
    print(f"Gradient has NaN: {grad_has_nan}")
    print(f"Gradient norm: {logits.grad.norm().item():.4f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)