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
                 num_labels: int):
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
        self.post_init()

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

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )