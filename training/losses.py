import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    
    论文: Focal Loss for Dense Object Detection
    适用场景: 类别分布不均衡的分类任务
    
    参数:
    - alpha: 类别权重，可以是float或tensor
    - gamma: 聚焦参数，默认2.0
    - reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss - 防止过拟合
    
    通过软化标签来提高模型泛化性能
    
    参数:
    - smoothing: 平滑参数，通常0.1
    - temperature: 温度参数，默认1.0
    """
    
    def __init__(self, smoothing=0.1, temperature=1.0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.temperature = temperature
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs / self.temperature, dim=-1)
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = (1 - self.smoothing) * targets_one_hot + \
                        self.smoothing / inputs.size(-1)
        
        loss = -torch.sum(targets_smooth * log_probs, dim=-1)
        return loss.mean()

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss - 提高特征表示质量
    
    论文: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    适用场景: 需要高质量特征表示的分类任务
    
    参数:
    - in_features: 输入特征维度
    - out_features: 输出类别数
    - s: 缩放因子，默认30.0
    - m: 角度边距，默认0.5
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
        
    def forward(self, inputs, targets):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, targets)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss - 对比学习
    
    论文: Supervised Contrastive Learning
    适用场景: 需要学习更好特征表示的任务
    
    参数:
    - temperature: 温度参数，默认0.07
    - contrast_mode: 'all' 或 'one'
    - base_temperature: 基础温度，默认0.07
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, labels=None, mask=None):
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
            
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross-Entropy Loss - 对噪声标签更鲁棒
    
    论文: Symmetric Cross-entropy for Robust Learning with Noisy Labels
    适用场景: 标签可能有噪声的分类任务
    
    参数:
    - alpha: CE loss的权重
    - beta: RCE loss的权重
    - num_classes: 类别数量
    """
    
    def __init__(self, alpha=1.0, beta=1.0, num_classes=101):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        # Standard Cross-Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean')
        
        # Reverse Cross-Entropy
        pred_softmax = F.softmax(inputs, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)
        
        targets_one_hot = torch.zeros_like(pred_softmax)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)
        
        rce_loss = -torch.sum(pred_softmax * torch.log(targets_one_hot), dim=1)
        rce_loss = rce_loss.mean()
        
        # Combine both losses
        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss

class MixupLoss(nn.Module):
    """
    Mixup Loss - 数据增强相关的损失函数
    
    论文: mixup: Beyond Empirical Risk Minimization
    适用场景: 提高模型泛化性能
    
    参数:
    - alpha: mixup参数，控制混合强度
    """
    
    def __init__(self, alpha=1.0):
        super(MixupLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, inputs, targets_a, targets_b, lam):
        """
        inputs: 模型预测
        targets_a, targets_b: 混合的两个标签
        lam: 混合比例
        """
        criterion = nn.CrossEntropyLoss()
        return lam * criterion(inputs, targets_a) + (1 - lam) * criterion(inputs, targets_b)

def create_loss_function(loss_type='cross_entropy', **kwargs):
    """
    损失函数工厂函数
    
    参数:
    - loss_type: 损失函数类型
    - **kwargs: 损失函数的参数
    
    返回:
    - 损失函数实例
    """
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        temperature = kwargs.get('temperature', 1.0)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, temperature=temperature)
    
    elif loss_type == 'arcface':
        in_features = kwargs.get('in_features', 768)
        out_features = kwargs.get('out_features', 101)
        s = kwargs.get('s', 30.0)
        m = kwargs.get('m', 0.5)
        return ArcFaceLoss(in_features=in_features, out_features=out_features, s=s, m=m)
    
    elif loss_type == 'supcon':
        temperature = kwargs.get('temperature', 0.07)
        return SupConLoss(temperature=temperature)
    
    elif loss_type == 'symmetric_ce':
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 1.0)
        num_classes = kwargs.get('num_classes', 101)
        return SymmetricCrossEntropy(alpha=alpha, beta=beta, num_classes=num_classes)
    
    elif loss_type == 'mixup':
        alpha = kwargs.get('alpha', 1.0)
        return MixupLoss(alpha=alpha)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# 推荐的损失函数配置
RECOMMENDED_LOSS_CONFIGS = {
    'food_classification_balanced': {
        'type': 'label_smoothing',
        'smoothing': 0.1,
        'temperature': 1.0
    },
    
    'food_classification_imbalanced': {
        'type': 'focal',
        'alpha': 1.0,
        'gamma': 2.0
    },
    
    'food_classification_high_quality': {
        'type': 'arcface',
        'in_features': 768,  # Qwen2.5-VL hidden size
        'out_features': 101,  # food101 classes
        's': 30.0,
        'm': 0.5
    },
    
    'food_classification_noisy_labels': {
        'type': 'symmetric_ce',
        'alpha': 1.0,
        'beta': 1.0,
        'num_classes': 101
    }
} 