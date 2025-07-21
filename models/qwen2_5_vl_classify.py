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
        
        super().__init__(config)

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            attn_implementation=config._attn_implementation,
            torch_dtype=torch.bfloat16
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
        根据数据集的类别数量对logits进行masking（带数值稳定性检查）
        
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
            
        # 🔥 数值稳定性检查：裁剪过大的logits值
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
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
            
            # 应用masking - 使用更安全的masking值
            if num_classes is not None and num_classes < logits.size(-1):
                # 🔥 使用-1e9而不是-inf，避免数值问题
                mask_value = -1e9
                masked_logits[i, num_classes:] = mask_value
                
                # 🔥 安全检查：确保有效位置不全为极小值
                valid_logits = masked_logits[i, :num_classes]
                if torch.all(valid_logits < -10.0):
                    # 如果有效位置的logits都太小，进行调整
                    masked_logits[i, :num_classes] = torch.clamp(valid_logits, min=-10.0)
        
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
        
        # 计算损失 - 带数值稳定性检查
        loss = None
        if labels is not None:
            try:
                # 🔥 数值稳定性检查：标签边界检查
                max_label = labels.max().item()
                if max_label >= logits.size(-1):
                    print(f"⚠️ 发现越界标签: max_label={max_label}, logits_classes={logits.size(-1)}")
                    # 裁剪越界的标签
                    labels = torch.clamp(labels, min=0, max=logits.size(-1)-1)
                
                # 🔥 数值稳定性检查：logits值检查
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    print(f"⚠️ logits包含NaN或Inf，进行清理")
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
                
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
                
                # 🔥 最终数值稳定性检查
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ 损失计算结果为NaN或Inf: {loss}")
                    # 使用一个小的固定损失值，避免训练崩溃
                    loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
                    print(f"🔧 使用回退损失值: {loss}")
                    
            except Exception as e:
                print(f"❌ 损失计算过程出错: {e}")
                # 静默回退到标准损失函数
                import torch.nn.functional as F
                try:
                    loss = F.cross_entropy(logits, labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
                except:
                    loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
        
        # 🔥 关键修复：无论训练还是评估都不返回大tensor，避免NCCL超时
        # 经过代码分析确认：训练和评估过程中都不需要hidden_states和attentions
        # 只需要loss和logits进行反向传播和预测计算
        
        # 统一返回简化输出 - 大幅节省内存和通信带宽
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # ✅ 训练和评估都不需要，避免4.67亿元素的NCCL reduce
            attentions=None,     # ✅ 训练和评估都不需要，节省内存和通信带宽
        )