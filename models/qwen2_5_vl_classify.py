import torch
import torch.nn as nn
from transformers import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    AutoConfig,
    AutoProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import Qwen2_5_VLForConditionalGeneration

class Qwen2_5_VLForImageClassification(Qwen2_5_VLPreTrainedModel):
    """
    Image classification model that runs the full multimodal -> language pipeline
    and applies a classification head on the final token representations.

    This class loads pretrained weights from a Qwen2.5-VL-ForConditionalGeneration checkpoint.
    """
    def __init__(self,
                 pretrained_model_name: str,
                 num_labels: int,
                 loss_config: dict = None):
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.num_labels = num_labels
        super().__init__(config)

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.model = base_model.model
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        text_cfg = config.get_text_config()
        hidden_size = text_cfg.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 设置损失函数
        self.loss_config = loss_config or {'type': 'cross_entropy'}
        self.loss_function = self._create_loss_function()
        
        self.post_init()
        
    def _create_loss_function(self):
        """创建损失函数 - 内联实现，避免import冲突"""
        
        loss_type = self.loss_config.get('type', 'cross_entropy')
        
        # 只在主进程打印损失函数信息
        try:
            from training.utils.distributed import is_dist_initialized, get_rank
            should_print = not is_dist_initialized() or get_rank() == 0
        except:
            should_print = True
        
        if should_print:
            print(f"🎯 创建损失函数: {loss_type}")
        
        try:
            if loss_type == 'cross_entropy':
                return nn.CrossEntropyLoss()
            
            elif loss_type == 'label_smoothing':
                # 内联实现LabelSmoothingCrossEntropy
                smoothing = self.loss_config.get('smoothing', 0.1)
                temperature = self.loss_config.get('temperature', 1.0)
                
                class LabelSmoothingCrossEntropy(nn.Module):
                    def __init__(self, smoothing=0.1, temperature=1.0):
                        super().__init__()
                        self.smoothing = smoothing
                        self.temperature = temperature
                        
                    def forward(self, inputs, targets):
                        import torch.nn.functional as F
                        log_probs = F.log_softmax(inputs / self.temperature, dim=-1)
                        targets_one_hot = torch.zeros_like(log_probs)
                        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                        
                        # Apply label smoothing
                        targets_smooth = (1 - self.smoothing) * targets_one_hot + \
                                        self.smoothing / inputs.size(-1)
                        
                        loss = -torch.sum(targets_smooth * log_probs, dim=-1)
                        return loss.mean()
                
                return LabelSmoothingCrossEntropy(smoothing=smoothing, temperature=temperature)
            
            elif loss_type == 'focal':
                # 内联实现FocalLoss
                alpha = self.loss_config.get('alpha', 1.0)
                gamma = self.loss_config.get('gamma', 2.0)
                
                class FocalLoss(nn.Module):
                    def __init__(self, alpha=1.0, gamma=2.0):
                        super().__init__()
                        self.alpha = alpha
                        self.gamma = gamma
                        
                    def forward(self, inputs, targets):
                        import torch.nn.functional as F
                        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                        pt = torch.exp(-ce_loss)
                        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                        return focal_loss.mean()
                
                return FocalLoss(alpha=alpha, gamma=gamma)
            
            else:
                # 不支持的损失函数类型，回退到CrossEntropyLoss
                if should_print:
                    print(f"⚠️  不支持的损失函数类型: {loss_type}，回退到CrossEntropyLoss")
                return nn.CrossEntropyLoss()
                
        except Exception as e:
            if should_print:
                print(f"❌ 创建损失函数失败: {e}")
                print(f"🔄 回退到标准CrossEntropyLoss")
            return nn.CrossEntropyLoss()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # 使用attention_mask来获取正确的序列长度（包含visual tokens + text tokens）
        if attention_mask is not None:
            # attention_mask覆盖完整序列（visual + text tokens）
            valid_lengths = attention_mask.sum(dim=1)  # 每个样本的有效长度
            last_positions = valid_lengths - 1  # 最后一个有效位置
            # 添加边界检查，确保索引不越界
            last_positions = torch.clamp(last_positions, min=0, max=hidden_states.size(1)-1)
            pooled = hidden_states[torch.arange(hidden_states.size(0)), last_positions]
        else:
            # 如果没有attention_mask，使用序列的最后位置
            pooled = hidden_states[:, -1, :]

        # 计算logits
        logits = self.classifier(pooled)
        
        # 计算损失
        loss = None
        if labels is not None:
            try:
                # 所有当前支持的损失函数都使用logits和标签
                loss = self.loss_function(logits, labels)
                    
            except Exception as e:
                print(f"❌ 损失函数调用失败: {e}")
                print(f"🔄 回退到标准CrossEntropyLoss")
                # 创建一个标准的CrossEntropyLoss作为回退
                fallback_loss = nn.CrossEntropyLoss()
                loss = fallback_loss(logits, labels)
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )