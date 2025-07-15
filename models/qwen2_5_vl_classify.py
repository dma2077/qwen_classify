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
    Supports multi-dataset functionality with logits masking.
    """
    def __init__(self,  
                 pretrained_model_name: str,
                 num_labels: int,
                 loss_config: dict = None,
                 dataset_configs: dict = None,
                 enable_logits_masking: bool = True):
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.num_labels = num_labels
        
        # 确保配置不会触发ForSequenceClassificationLoss
        config.problem_type = None  # 重置problem_type
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        
        # 配置信息已设置完成，无需输出
        
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
        
        # 设置损失函数配置 - 不创建loss_function对象，避免继承问题
        self.loss_config = loss_config or {'type': 'cross_entropy'}
        
        # 多数据集配置
        self.dataset_configs = dataset_configs or {}
        self.enable_logits_masking = enable_logits_masking
        
        self.post_init()

    def _apply_logits_masking(self, logits, dataset_names=None, num_classes_list=None):
        """
        根据数据集的类别数量对logits进行masking
        
        Args:
            logits: [batch_size, num_labels] 的logits tensor
            dataset_names: 每个样本的数据集名称列表
            num_classes_list: 每个样本对应数据集的类别数量列表
        
        Returns:
            masked_logits: masking后的logits tensor
        """
        if not self.enable_logits_masking:
            return logits
            
        if dataset_names is None and num_classes_list is None:
            return logits
            
        masked_logits = logits.clone()
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            num_classes = None
            
            # 优先使用直接提供的num_classes
            if num_classes_list is not None and i < len(num_classes_list):
                num_classes = num_classes_list[i]
            
            # 如果没有直接提供，从dataset_configs中获取
            if num_classes is None and dataset_names is not None and i < len(dataset_names):
                dataset_name = dataset_names[i]
                dataset_config = self.dataset_configs.get(dataset_name, {})
                num_classes = dataset_config.get("num_classes", None)
            
            # 应用masking
            if num_classes is not None and num_classes < logits.size(-1):
                # 将超出数据集类别范围的logits设为很小的值（相当于mask掉）
                masked_logits[i, num_classes:] = float('-inf')
        
        return masked_logits

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor = None,
        dataset_names: list = None,
        num_classes_list: list = None,
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
        
        # 应用logits masking
        if self.enable_logits_masking:
            logits = self._apply_logits_masking(logits, dataset_names, num_classes_list)
        
        # 计算损失 - 直接在forward中创建，避免继承关系问题
        loss = None
        if labels is not None:
            try:
                # 不使用self.loss_function，直接在这里创建损失函数
                loss_type = self.loss_config.get('type', 'cross_entropy')
                
                if loss_type == 'label_smoothing':
                    smoothing = self.loss_config.get('smoothing', 0.1)
                    temperature = self.loss_config.get('temperature', 1.0)
                    
                    # 内联Label Smoothing实现
                    import torch.nn.functional as F
                    log_probs = F.log_softmax(logits / temperature, dim=-1)
                    targets_one_hot = torch.zeros_like(log_probs)
                    targets_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    
                    # Apply label smoothing
                    targets_smooth = (1 - smoothing) * targets_one_hot + smoothing / logits.size(-1)
                    loss = -torch.sum(targets_smooth * log_probs, dim=-1).mean()
                        
                elif loss_type == 'focal':
                    alpha = self.loss_config.get('alpha', 1.0)
                    gamma = self.loss_config.get('gamma', 2.0)
                    
                    # 内联Focal Loss实现
                    import torch.nn.functional as F
                    ce_loss = F.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
                        
                else:
                    # 标准CrossEntropyLoss
                    import torch.nn.functional as F
                    loss = F.cross_entropy(logits, labels)
                    
            except Exception as e:
                # 静默回退到标准损失函数
                import torch.nn.functional as F
                loss = F.cross_entropy(logits, labels)
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )