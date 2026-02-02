from mindspore import nn, ops

class HingeLoss(nn.Cell):
    """自定义 Hinge Loss"""
    
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean', 'sum' 或 'none'
        """
        super(HingeLoss, self).__init__()
        self.reduction = reduction
    
    def construct(self, logits, labels, weights):
        """
        Args:
            logits: 模型输出，形状 (batch_size,) 或 (batch_size, 1)
            labels: 真实标签，值为 -1 或 1
            weights: 样本权重，形状 (batch_size,)
        
        Returns:
            损失值
        """
        # 展平维度，确保都是 (batch_size,)
        logits = logits.view(-1)
        labels = labels.view(-1)
        weights = weights.view(-1)
        
        # 计算 hinge loss: max(0, 1 - y * y_pred)
        per_sample_loss = ops.maximum(0.0, 1.0 - labels * logits)
        loss = ops.sum(per_sample_loss * weights)
        
        return loss