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
        
        # 确保配置不会触发ForSequenceClassificationLoss
        config.problem_type = None  # 重置problem_type
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        
        # 只在主进程打印配置信息
        try:
            from training.utils.distributed import is_dist_initialized, get_rank
            should_print = not is_dist_initialized() or get_rank() == 0
        except:
            should_print = True
        
        if should_print:
            print(f"🔍 模型配置:")
            print(f"   num_labels: {config.num_labels}")
            print(f"   problem_type: {getattr(config, 'problem_type', 'None')}")
            print(f"   config type: {type(config)}")
            print(f"   loss_config: {loss_config}")
        
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
        # 注释掉，直接在forward中内联计算损失
        # self.loss_function = self._create_loss_function()
        
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

        # 计算logits
        logits = self.classifier(pooled)
        
        # 计算损失 - 直接在forward中创建，避免继承关系问题
        loss = None
        if labels is not None:
            # 调试信息
            try:
                from training.utils.distributed import is_dist_initialized, get_rank
                should_print = not is_dist_initialized() or get_rank() == 0
            except:
                should_print = True
            
            if should_print:
                print(f"🔍 Forward调用 - logits shape: {logits.shape}")
                print(f"🔍 Forward调用 - labels shape: {labels.shape}")
            
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
                    
                    if should_print:
                        print(f"✅ Label Smoothing损失计算成功: {loss.item():.4f}")
                        
                elif loss_type == 'focal':
                    alpha = self.loss_config.get('alpha', 1.0)
                    gamma = self.loss_config.get('gamma', 2.0)
                    
                    # 内联Focal Loss实现
                    import torch.nn.functional as F
                    ce_loss = F.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
                    
                    if should_print:
                        print(f"✅ Focal Loss损失计算成功: {loss.item():.4f}")
                        
                else:
                    # 标准CrossEntropyLoss
                    import torch.nn.functional as F
                    loss = F.cross_entropy(logits, labels)
                    
                    if should_print:
                        print(f"✅ CrossEntropy损失计算成功: {loss.item():.4f}")
                    
            except Exception as e:
                if should_print:
                    print(f"❌ 损失函数计算失败: {e}")
                    print(f"🔄 回退到标准F.cross_entropy")
                # 最终回退
                import torch.nn.functional as F
                loss = F.cross_entropy(logits, labels)
                if should_print:
                    print(f"✅ 回退损失函数计算成功: {loss.item():.4f}")
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )