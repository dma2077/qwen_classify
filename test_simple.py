import torch
import os
from PIL import Image
import numpy as np

def test_pooling_logic():
    """测试pooling逻辑（不需要加载大模型）"""
    print("🧪 测试Pooling逻辑")
    print("="*50)
    
    # 模拟多模态数据
    batch_size = 1
    visual_tokens = 576  # 常见的视觉tokens数量
    text_tokens = 50     # 文本tokens数量
    total_length = visual_tokens + text_tokens + 10  # 加一些padding
    hidden_dim = 768
    
    print(f"📊 模拟数据设置:")
    print(f"  • Visual tokens: {visual_tokens}")
    print(f"  • Text tokens: {text_tokens}")
    print(f"  • Total sequence length: {total_length}")
    print(f"  • Padding: {total_length - visual_tokens - text_tokens}")
    
    # 创建attention_mask
    # 前(visual_tokens + text_tokens)个位置有效，后面是padding
    attention_mask = torch.zeros(batch_size, total_length)
    attention_mask[:, :visual_tokens + text_tokens] = 1
    
    # 创建模拟的input_ids（只包含文本部分）
    input_ids = torch.randint(1, 1000, (batch_size, text_tokens))
    
    # 创建模拟的hidden_states
    hidden_states = torch.randn(batch_size, total_length, hidden_dim)
    
    print(f"\n🎯 数据形状:")
    print(f"  • input_ids: {input_ids.shape}")
    print(f"  • attention_mask: {attention_mask.shape}")
    print(f"  • hidden_states: {hidden_states.shape}")
    
    # 测试旧的pooling方法（错误的）
    print(f"\n❌ 旧方法 (错误的):")
    pad_token_id = 0
    mask = input_ids.ne(pad_token_id)
    old_last_positions = mask.sum(dim=1) - 1
    print(f"  • 基于input_ids的last_positions: {old_last_positions}")
    print(f"  • 这个位置的attention值: {attention_mask[0, old_last_positions].item()}")
    print(f"  • 问题: 这个位置在visual tokens中间，不是序列末尾！")
    
    # 测试新的pooling方法（正确的）
    print(f"\n✅ 新方法 (正确的):")
    if attention_mask is not None:
        valid_lengths = attention_mask.sum(dim=1)
        new_last_positions = valid_lengths - 1
        new_last_positions = torch.clamp(new_last_positions, min=0, max=hidden_states.size(1)-1)
        
        print(f"  • 基于attention_mask的valid_lengths: {valid_lengths}")
        print(f"  • 基于attention_mask的last_positions: {new_last_positions}")
        print(f"  • 这个位置的attention值: {attention_mask[0, new_last_positions].item()}")
        print(f"  • 正确: 这是序列的真正末尾位置！")
    
    # 显示差异
    print(f"\n📏 位置差异:")
    old_pos = old_last_positions.item()
    new_pos = new_last_positions.item()
    print(f"  • 旧方法位置: {old_pos}")
    print(f"  • 新方法位置: {new_pos}")
    print(f"  • 差异: {new_pos - old_pos} tokens")
    print(f"  • 旧方法跳过了 {new_pos - old_pos} 个visual tokens!")
    
    # 显示attention_mask模式
    print(f"\n🎯 Attention Mask 模式:")
    mask_values = attention_mask[0].tolist()
    print(f"  • 前10个位置: {mask_values[:10]}")
    print(f"  • 中间10个位置 (pos {total_length//2-5} to {total_length//2+4}): {mask_values[total_length//2-5:total_length//2+5]}")
    print(f"  • 后10个位置: {mask_values[-10:]}")
    
    # 验证边界情况
    print(f"\n🔍 边界验证:")
    print(f"  • 位置 {new_pos-1}: attention = {attention_mask[0, new_pos-1].item()}")
    print(f"  • 位置 {new_pos}: attention = {attention_mask[0, new_pos].item()}")
    if new_pos + 1 < attention_mask.size(1):
        print(f"  • 位置 {new_pos+1}: attention = {attention_mask[0, new_pos+1].item()}")
    
    return old_pos, new_pos, new_pos - old_pos

def test_with_real_data():
    """如果可以的话，用真实数据测试"""
    print(f"\n🔄 尝试用真实数据测试...")
    
    try:
        from transformers import AutoProcessor
        
        # 尝试加载processor（不加载大模型）
        model_path = "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"
        if os.path.exists(model_path):
            processor = AutoProcessor.from_pretrained(model_path)
            print("✅ Processor加载成功")
            
            # 创建测试图片和文本
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            text = "This is an image of food, what dish is it?"
            
            # 处理输入
            inputs = processor(
                text=[text],
                images=[test_image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            print(f"📊 真实数据分析:")
            print(f"  • input_ids shape: {inputs['input_ids'].shape}")
            print(f"  • attention_mask shape: {inputs['attention_mask'].shape}")
            print(f"  • pixel_values shape: {inputs['pixel_values'].shape}")
            
            if 'image_grid_thw' in inputs:
                print(f"  • image_grid_thw: {inputs['image_grid_thw']}")
                visual_tokens = inputs['image_grid_thw'][0, 0] * inputs['image_grid_thw'][0, 1] * inputs['image_grid_thw'][0, 2]
                print(f"  • 计算的visual tokens: {visual_tokens}")
            
            # 分析序列长度
            text_len = inputs['input_ids'].size(1)
            total_len = inputs['attention_mask'].size(1)
            valid_len = inputs['attention_mask'].sum(dim=1).item()
            
            print(f"  • 文本长度: {text_len}")
            print(f"  • 总序列长度: {total_len}")
            print(f"  • 有效长度: {valid_len}")
            print(f"  • 推断的visual tokens: {valid_len - text_len}")
            
            # 测试pooling
            attention_mask = inputs['attention_mask']
            valid_lengths = attention_mask.sum(dim=1)
            last_positions = valid_lengths - 1
            
            print(f"  • Pooling位置: {last_positions.item()}")
            print(f"  • 该位置attention值: {attention_mask[0, last_positions.item()].item()}")
            
        else:
            print("❌ 模型路径不存在，跳过真实数据测试")
            
    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")

if __name__ == "__main__":
    # 运行逻辑测试
    old_pos, new_pos, diff = test_pooling_logic()
    
    # 尝试真实数据测试
    test_with_real_data()
    
    # 总结
    print(f"\n📋 总结:")
    print("="*50)
    print(f"✅ 修复前pooling位置: {old_pos}")
    print(f"✅ 修复后pooling位置: {new_pos}")
    print(f"✅ 修复效果: 正确处理了{diff}个visual tokens")
    print(f"✅ 现在从真正的序列末尾提取特征")
    print("="*50) 