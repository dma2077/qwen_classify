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
        """创建损失函数"""
        from training.losses import create_loss_function
        
        loss_type = self.loss_config.get('type', 'cross_entropy')
        
        # 为ArcFace损失传入正确的特征维度
        if loss_type == 'arcface':
            text_cfg = self.config.get_text_config()
            hidden_size = text_cfg.hidden_size
            self.loss_config.update({
                'in_features': hidden_size,
                'out_features': self.config.num_labels
            })
            
        return create_loss_function(loss_type, **self.loss_config)

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
            loss_type = self.loss_config.get('type', 'cross_entropy')
            
            if loss_type == 'arcface':
                # ArcFace损失需要原始特征和标签
                loss = self.loss_function(pooled, labels)
            elif loss_type == 'supcon':
                # SupCon损失需要特征和标签（需要特殊的数据格式）
                # 这里简化处理，实际使用时需要准备对比学习的数据格式
                features = pooled.unsqueeze(1)  # [batch_size, 1, hidden_size]
                loss = self.loss_function(features, labels)
            else:
                # 标准损失函数使用logits和标签
                loss = self.loss_function(logits, labels)
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )