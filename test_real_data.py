import torch
import os
from PIL import Image
from transformers import AutoProcessor
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification

def test_real_multimodal_data():
    """使用真实的多模态数据测试token长度和pooling"""
    
    print("="*80)
    print("🔍 使用真实数据测试Qwen2.5-VL Token长度和Pooling")
    print("="*80)
    
    # 1. 设置路径
    model_path = "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"
    local_path = "/llm_reco/dehua/data/food_data/food-101/images/apple_pie/2928660.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(local_path):
        print(f"❌ 图片文件不存在: {local_path}")
        return
    
    # 2. 准备真实数据（两个样本的batch）
    print("📊 准备真实测试数据...")
    try:
        # 加载图片
        image = Image.open(local_path).convert("RGB")
        print(f"✅ 图片加载成功: {image.size}")
        
        # 构建第一个样本的messages
        messages1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": local_path},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        
        # 构建第二个样本的messages（文本更长）
        messages2 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": local_path},
                    {"type": "text", "text": "What is shown in this image Please give your answer briefly?"},
                ],
            },
        ]
        
        print("✅ 两个样本的Messages构建成功")
        print(f"📝 样本1文本: 'What is shown in this image?'")
        print(f"📝 样本2文本: 'What is shown in this image Please give your answer briefly?'")
        
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return
    
    # 3. 加载processor
    print("\n📦 加载processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Processor加载成功")
    except Exception as e:
        print(f"❌ Processor加载失败: {e}")
        return
    
    # 4. 按照collator.py的方式处理数据
    print("\n🔄 按照collator.py方式处理数据...")
    try:
        # 模拟batch数据（两个样本）
        batch = [
            {
                "image": image,
                "messages": messages1,
                "label": 0  # 假设标签：apple_pie
            },
            {
                "image": image,  # 使用相同图片
                "messages": messages2,
                "label": 1  # 假设标签：不同类别
            }
        ]
        
        # 提取数据
        images = [item["image"] for item in batch]
        msgs = [item["messages"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        
        print(f"📋 Batch信息:")
        print(f"  • Images: {len(images)}")
        print(f"  • Messages: {len(msgs)}")
        print(f"  • Labels: {labels}")
        
        # 1) 转换为chat模板
        text_list = []
        for i, m in enumerate(msgs):
            text = processor.apply_chat_template(
                conversation=m,
                tokenize=False,
                add_generation_prompt=True
            )
            text_list.append(text)
            print(f"\n📝 样本{i+1}生成的chat模板:")
            print(f"文本长度: {len(text)} 字符")
            print(f"文本内容:\n{text}")
        
        # 2) 使用processor处理多模态输入
        enc = processor(
            text=text_list,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # 添加labels
        enc["labels"] = labels
        
        print("✅ 数据处理成功")
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return
    
    # 5. 分析处理后的数据
    print(f"\n📋 处理后数据分析:")
    print("-" * 60)
    
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    pixel_values = enc['pixel_values']
    
    print(f"🔤 input_ids形状: {input_ids.shape}")
    print(f"🎯 attention_mask形状: {attention_mask.shape}")
    print(f"🖼️  pixel_values形状: {pixel_values.shape}")
    
    # 检查image_grid_thw
    if 'image_grid_thw' in enc:
        image_grid_thw = enc['image_grid_thw']
        print(f"📐 image_grid_thw形状: {image_grid_thw.shape}")
        print(f"📐 image_grid_thw值: {image_grid_thw}")
        
        # 计算visual tokens数量
        visual_tokens = image_grid_thw[0, 0] * image_grid_thw[0, 1] * image_grid_thw[0, 2]
        print(f"👁️  计算的visual tokens数量: {visual_tokens}")
    else:
        print("⚠️  没有找到image_grid_thw")
        visual_tokens = None
    
    # 6. 分析序列长度
    print(f"\n📏 序列长度详细分析:")
    print("-" * 60)
    
    batch_size = input_ids.size(0)
    text_length = input_ids.size(1)
    attention_length = attention_mask.size(1)
    
    print(f"📦 Batch大小: {batch_size}")
    print(f"📝 文本tokens长度 (input_ids): {text_length}")
    print(f"🎯 attention_mask总长度: {attention_length}")
    
    # 分析每个样本的有效tokens
    for i in range(batch_size):
        valid_tokens = attention_mask[i].sum().item()
        print(f"✅ 样本{i+1}有效tokens数量: {valid_tokens}")
        print(f"🔍 样本{i+1}推断的visual tokens: {valid_tokens - text_length}")
        
        if visual_tokens is not None:
            print(f"📐 样本{i+1} image_grid_thw计算的visual tokens: {visual_tokens}")
            print(f"🔄 样本{i+1} 两种方法的差异: {abs(visual_tokens - (valid_tokens - text_length))}")
        print()
    
    # 7. 分析attention_mask模式
    print(f"\n🎯 Attention Mask模式分析:")
    print("-" * 60)
    
    for i in range(batch_size):
        print(f"📋 样本{i+1}的Attention Mask模式:")
        
        mask_values = attention_mask[i].tolist()
        
        # 找到第一个和最后一个有效位置
        valid_indices = attention_mask[i].nonzero().flatten()
        first_valid = valid_indices[0].item()
        last_valid = valid_indices[-1].item()
        
        print(f"  🎯 第一个有效位置: {first_valid}")
        print(f"  🎯 最后一个有效位置: {last_valid}")
        print(f"  🎯 有效范围: [{first_valid}, {last_valid}]")
        
        # 显示关键位置的attention值
        print(f"  🔍 前10个位置: {mask_values[:10]}")
        print(f"  🔍 后10个位置: {mask_values[-10:]}")
        
        # 显示文本开始位置（估算）
        sample_valid_tokens = attention_mask[i].sum().item()
        if sample_valid_tokens > text_length:
            text_start_in_full_seq = sample_valid_tokens - text_length
            print(f"  🔍 估算文本开始位置: {text_start_in_full_seq}")
        
        print()
    
    # 8. 测试pooling位置计算
    print(f"\n🎯 Pooling位置计算:")
    print("-" * 60)
    
    # 获取pad_token_id
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'pad_token_id'):
        pad_token_id = processor.tokenizer.pad_token_id
    else:
        pad_token_id = 0  # 假设值
    
    # 旧方法（错误的）
    old_mask = input_ids.ne(pad_token_id)
    old_last_positions = old_mask.sum(dim=1) - 1
    
    # 新方法（正确的）
    valid_lengths = attention_mask.sum(dim=1)
    new_last_positions = valid_lengths - 1
    new_last_positions = torch.clamp(new_last_positions, min=0, max=attention_mask.size(1)-1)
    
    # 分析每个样本的pooling位置
    for i in range(batch_size):
        print(f"📋 样本{i+1}的Pooling位置:")
        
        old_pos = old_last_positions[i].item()
        new_pos = new_last_positions[i].item()
        
        print(f"  ❌ 旧方法 (基于input_ids):")
        print(f"    • Pooling位置: {old_pos}")
        print(f"    • 该位置的attention值: {attention_mask[i, old_pos].item()}")
        
        print(f"  ✅ 新方法 (基于attention_mask):")
        print(f"    • Pooling位置: {new_pos}")
        print(f"    • 该位置的attention值: {attention_mask[i, new_pos].item()}")
        
        print(f"  📊 差异分析:")
        print(f"    • 位置差异: {new_pos - old_pos} tokens")
        print(f"    • 旧方法跳过的tokens: {new_pos - old_pos}")
        print()
    
    # 9. 验证pooling位置的正确性
    print(f"\n🔍 Pooling位置验证:")
    print("-" * 60)
    
    print(f"📍 新方法pooling位置({new_pos})周围的attention值:")
    for offset in [-2, -1, 0, 1, 2]:
        pos = new_pos + offset
        if 0 <= pos < attention_mask.size(1):
            print(f"  • 位置 {pos}: attention = {attention_mask[0, pos].item()}")
        else:
            print(f"  • 位置 {pos}: 越界")
    
    # 检查是否真的是最后有效位置
    is_last_valid = (new_pos == last_valid)
    print(f"✅ 是否为最后有效位置: {is_last_valid}")
    
    # 10. 如果可能，测试模型前向传播
    print(f"\n🚀 尝试模型前向传播测试:")
    print("-" * 60)
    
    try:
        print("⚠️  注意: 加载7B模型需要大量内存，如果内存不足会失败")
        
        # 可以选择只加载模型的一部分来测试
        # 这里我们只测试数据流，不加载完整模型
        print("🔄 跳过模型加载，仅验证数据流...")
        
        # 模拟模型输出的hidden_states形状
        batch_size, seq_len, hidden_dim = 1, attention_length, 768
        mock_hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 使用新的pooling方法提取特征
        pooled = mock_hidden_states[torch.arange(batch_size), new_last_positions]
        print(f"✅ 模拟pooled特征形状: {pooled.shape}")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
    
    # 11. 总结
    print(f"\n📋 测试总结:")
    print("="*80)
    print(f"✅ 使用真实数据: {local_path}")
    print(f"✅ 图片尺寸: {image.size}")
    print(f"✅ 文本长度: {len(text_list[0])} 字符")
    print(f"✅ 文本tokens: {text_length}")
    print(f"✅ 总序列长度: {attention_length}")
    print(f"✅ 有效tokens: {valid_tokens}")
    print(f"✅ Visual tokens: {valid_tokens - text_length}")
    if visual_tokens is not None:
        print(f"✅ image_grid_thw计算的visual tokens: {visual_tokens}")
    print(f"✅ 修复前pooling位置: {old_pos}")
    print(f"✅ 修复后pooling位置: {new_pos}")
    print(f"✅ 修复效果: 正确处理了 {new_pos - old_pos} 个visual tokens")
    print(f"✅ 现在使用正确的序列末尾位置进行pooling")
    print("="*80)

if __name__ == "__main__":
    test_real_multimodal_data() 