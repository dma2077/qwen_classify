import torch
import os
from PIL import Image
from transformers import AutoProcessor
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification

def test_token_lengths_and_pooling():
    """测试token长度和pooling位置"""
    
    # 配置
    model_path = "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"  # 根据你的路径调整
    test_image_path = "/llm_reco/dehua/code/qwen_classify/data/dataset_food_label/images/test_image.jpg"  # 请提供一个测试图片路径
    
    print("="*80)
    print("🔍 测试Qwen2.5-VL模型的Token长度和Pooling位置")
    print("="*80)
    
    # 1. 加载模型和processor
    print("📦 加载模型和processor...")
    try:
        model = Qwen2_5_VLForImageClassification(
            pretrained_model_name=model_path,
            num_labels=101
        )
        processor = AutoProcessor.from_pretrained(model_path)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 准备测试数据
    print("\n📊 准备测试数据...")
    
    # 如果没有测试图片，创建一个dummy图片
    if not os.path.exists(test_image_path):
        print(f"⚠️  测试图片不存在，创建一个dummy图片")
        # 创建一个简单的RGB图片
        import numpy as np
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_image = dummy_image
    else:
        test_image = Image.open(test_image_path).convert("RGB")
    
    # 准备messages
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "This is an image of food, what dish is it?"}
            ]
        }
    ]
    
    # 3. 处理输入
    print("🔄 处理输入数据...")
    try:
        # 转换为chat模板
        text = processor.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"📝 生成的文本模板:\n{text}")
        print(f"📏 文本长度: {len(text)} 字符")
        
        # 使用processor处理
        inputs = processor(
            text=[text],
            images=[test_image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        print("✅ 输入处理成功")
        
    except Exception as e:
        print(f"❌ 输入处理失败: {e}")
        return
    
    # 4. 分析输入数据
    print("\n📋 输入数据分析:")
    print("-" * 60)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    pixel_values = inputs['pixel_values']
    
    print(f"🔤 input_ids 形状: {input_ids.shape}")
    print(f"🎯 attention_mask 形状: {attention_mask.shape}")
    print(f"🖼️  pixel_values 形状: {pixel_values.shape}")
    
    if 'image_grid_thw' in inputs:
        image_grid_thw = inputs['image_grid_thw']
        print(f"📐 image_grid_thw 形状: {image_grid_thw.shape}")
        print(f"📐 image_grid_thw 值: {image_grid_thw}")
        
        # 计算visual tokens数量
        visual_tokens = image_grid_thw[0, 0] * image_grid_thw[0, 1] * image_grid_thw[0, 2]
        print(f"👁️  估算的visual tokens数量: {visual_tokens}")
    else:
        print("⚠️  没有找到image_grid_thw参数")
    
    # 5. 分析序列长度
    print(f"\n📏 序列长度分析:")
    print("-" * 60)
    
    text_length = input_ids.size(1)
    attention_length = attention_mask.size(1)
    valid_tokens = attention_mask.sum(dim=1).item()
    
    print(f"📝 文本tokens长度 (input_ids): {text_length}")
    print(f"🎯 attention_mask长度: {attention_length}")
    print(f"✅ 有效tokens数量: {valid_tokens}")
    print(f"🔍 推断的visual tokens: {valid_tokens - text_length}")
    
    # 显示attention_mask的模式
    print(f"\n🎯 Attention Mask 分析:")
    print("-" * 60)
    mask_values = attention_mask[0].tolist()
    print(f"前20个位置: {mask_values[:20]}")
    if len(mask_values) > 40:
        print(f"中间20个位置: {mask_values[len(mask_values)//2-10:len(mask_values)//2+10]}")
    print(f"后20个位置: {mask_values[-20:]}")
    
    # 找到第一个和最后一个有效位置
    first_valid = attention_mask[0].nonzero()[0].item()
    last_valid = attention_mask[0].nonzero()[-1].item()
    print(f"🎯 第一个有效位置: {first_valid}")
    print(f"🎯 最后一个有效位置: {last_valid}")
    
    # 6. 运行模型前向传播
    print(f"\n🚀 运行模型前向传播:")
    print("-" * 60)
    
    with torch.no_grad():
        # 获取模型内部的hidden states
        model_outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
            **{k: v for k, v in inputs.items() if k not in ['input_ids', 'attention_mask', 'pixel_values']}
        )
        
        hidden_states = model_outputs.last_hidden_state
        print(f"🧠 Hidden states 形状: {hidden_states.shape}")
        
        # 计算pooling位置
        if attention_mask is not None:
            valid_lengths = attention_mask.sum(dim=1)
            last_positions = valid_lengths - 1
            last_positions = torch.clamp(last_positions, min=0, max=hidden_states.size(1)-1)
            
            print(f"📍 计算的valid_lengths: {valid_lengths}")
            print(f"📍 计算的last_positions: {last_positions}")
            print(f"📍 实际使用的pooling位置: {last_positions.item()}")
            
            # 提取pooled特征
            pooled = hidden_states[torch.arange(hidden_states.size(0)), last_positions]
            print(f"🎯 Pooled特征形状: {pooled.shape}")
            
            # 验证：显示pooling位置前后的attention值
            pos = last_positions.item()
            print(f"\n🔍 Pooling位置验证:")
            print(f"位置 {pos-2}: attention_mask = {attention_mask[0, pos-2].item() if pos >= 2 else 'N/A'}")
            print(f"位置 {pos-1}: attention_mask = {attention_mask[0, pos-1].item() if pos >= 1 else 'N/A'}")
            print(f"位置 {pos}: attention_mask = {attention_mask[0, pos].item()}")
            print(f"位置 {pos+1}: attention_mask = {attention_mask[0, pos+1].item() if pos < attention_mask.size(1)-1 else 'N/A'}")
            print(f"位置 {pos+2}: attention_mask = {attention_mask[0, pos+2].item() if pos < attention_mask.size(1)-2 else 'N/A'}")
            
        else:
            pooled = hidden_states[:, -1, :]
            print(f"⚠️  没有attention_mask，使用最后位置")
            print(f"🎯 Pooled特征形状: {pooled.shape}")
    
    # 7. 运行完整的分类模型
    print(f"\n🎯 运行完整分类模型:")
    print("-" * 60)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **{k: v for k, v in inputs.items() if k not in ['input_ids', 'attention_mask', 'pixel_values']}
        )
        
        print(f"📊 Logits 形状: {outputs.logits.shape}")
        print(f"🏆 预测类别: {outputs.logits.argmax(dim=-1).item()}")
        print(f"📈 最大概率: {torch.softmax(outputs.logits, dim=-1).max().item():.4f}")
    
    # 8. 总结
    print(f"\n📋 测试总结:")
    print("="*80)
    print(f"✅ 文本tokens数量: {text_length}")
    print(f"✅ 总序列长度: {attention_length}")
    print(f"✅ 有效tokens数量: {valid_tokens}")
    print(f"✅ Visual tokens数量: {valid_tokens - text_length}")
    print(f"✅ Pooling位置: 第{last_positions.item()}个位置 (0-indexed)")
    print(f"✅ 是否使用最后有效位置: {'是' if last_positions.item() == valid_tokens - 1 else '否'}")
    print("="*80)

def create_simple_test():
    """创建一个简化的测试，不需要大模型"""
    print("\n🧪 简化测试 (不加载大模型):")
    print("-" * 60)
    
    # 模拟数据
    batch_size = 2
    total_seq_len = 100
    text_len = 30
    
    # 创建模拟的attention_mask
    # 前70个位置有效（visual tokens + text tokens），后30个是padding
    attention_mask = torch.zeros(batch_size, total_seq_len)
    attention_mask[:, :70] = 1  # 前70个位置有效
    
    # 模拟hidden_states
    hidden_dim = 768
    hidden_states = torch.randn(batch_size, total_seq_len, hidden_dim)
    
    print(f"📊 模拟数据:")
    print(f"  • batch_size: {batch_size}")
    print(f"  • total_seq_len: {total_seq_len}")
    print(f"  • text_len: {text_len}")
    print(f"  • hidden_dim: {hidden_dim}")
    
    # 测试pooling逻辑
    if attention_mask is not None:
        valid_lengths = attention_mask.sum(dim=1)
        last_positions = valid_lengths - 1
        last_positions = torch.clamp(last_positions, min=0, max=hidden_states.size(1)-1)
        
        print(f"\n🎯 Pooling计算:")
        print(f"  • valid_lengths: {valid_lengths}")
        print(f"  • last_positions: {last_positions}")
        
        # 提取特征
        pooled = hidden_states[torch.arange(hidden_states.size(0)), last_positions]
        print(f"  • pooled shape: {pooled.shape}")
        
        # 验证
        for i in range(batch_size):
            pos = last_positions[i].item()
            print(f"  • 样本{i}: 从位置{pos}提取特征，attention值={attention_mask[i, pos].item()}")

if __name__ == "__main__":
    # 运行测试
    try:
        test_token_lengths_and_pooling()
    except Exception as e:
        print(f"❌ 完整测试失败: {e}")
        print("🔄 运行简化测试...")
        create_simple_test() 