#!/usr/bin/env python3
"""
批次大小设计原理说明
"""

def explain_batch_size_design():
    """解释为什么Training和Evaluation使用不同的batch size"""
    
    print("🎯 Batch Size设计原理解释")
    print("=" * 60)
    
    # 配置示例
    micro_batch_size_per_gpu = 2
    num_gpus = 8
    gradient_accumulation_steps = 4
    train_batch_size = 64  # DeepSpeed配置中的总batch size
    
    print("📊 示例配置:")
    print(f"  • micro_batch_size_per_gpu: {micro_batch_size_per_gpu}")
    print(f"  • num_gpus: {num_gpus}")
    print(f"  • gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  • train_batch_size (DeepSpeed): {train_batch_size}")
    print()
    
    # Training DataLoader
    train_dataloader_batch = micro_batch_size_per_gpu
    print("🚂 Training DataLoader:")
    print(f"  • DataLoader batch_size: {train_dataloader_batch}")
    print(f"  • 每个GPU每次处理: {train_dataloader_batch} 样本")
    print(f"  • 梯度累积前总处理: {train_dataloader_batch} × {num_gpus} = {train_dataloader_batch * num_gpus} 样本")
    print(f"  • 梯度累积步数: {gradient_accumulation_steps}")
    print(f"  • 有效批次大小: {train_dataloader_batch} × {num_gpus} × {gradient_accumulation_steps} = {train_dataloader_batch * num_gpus * gradient_accumulation_steps} 样本")
    print("  💡 DeepSpeed会自动累积梯度，直到达到有效批次大小才更新参数")
    print()
    
    # Evaluation DataLoader  
    eval_dataloader_batch = micro_batch_size_per_gpu * num_gpus
    print("🎯 Evaluation DataLoader:")
    print(f"  • DataLoader batch_size: {eval_dataloader_batch}")
    print(f"  • 每次评估处理: {eval_dataloader_batch} 样本")
    print(f"  • 无梯度累积，直接计算结果")
    print(f"  • 相当于gradient_accumulation_steps=1时的训练批次大小")
    print()
    
    print("🔍 设计原理:")
    print("1. 🚂 Training设计:")
    print("   • 使用小的micro batch避免显存爆炸")
    print("   • DeepSpeed自动处理梯度累积，保证训练效果")
    print("   • 有效批次大小与配置的train_batch_size一致")
    print()
    print("2. 🎯 Evaluation设计:")
    print("   • 使用中等大小batch，平衡速度和内存")
    print("   • 避免使用完整的train_batch_size(64)导致OOM")
    print("   • 保持统计意义，提供准确的评估结果")
    print()
    
    print("⚠️  如果评估也使用train_batch_size:")
    print(f"   • 需要{train_batch_size}个样本的显存")
    print(f"   • 可能导致OOM，特别是在大模型上")
    print(f"   • 评估速度可能更慢（大批次处理）")
    print()
    
    print("✅ 当前设计的优势:")
    print(f"   • Training: 内存友好({train_dataloader_batch}/GPU) + 大有效批次({train_batch_size})")
    print(f"   • Evaluation: 平衡的批次大小({eval_dataloader_batch}) + 快速评估")
    print(f"   • 避免OOM风险，提高训练稳定性")
    print("=" * 60)

def verify_calculation():
    """验证批次大小计算是否正确"""
    print("\n🧮 计算验证:")
    print("=" * 40)
    
    # 从DeepSpeed配置反推GPU数量的公式
    train_batch_size = 64
    micro_batch_size_per_gpu = 2  
    gradient_accumulation_steps = 4
    
    calculated_num_gpus = train_batch_size // (micro_batch_size_per_gpu * gradient_accumulation_steps)
    print(f"从DeepSpeed配置计算GPU数量:")
    print(f"  {train_batch_size} ÷ ({micro_batch_size_per_gpu} × {gradient_accumulation_steps}) = {calculated_num_gpus}")
    
    eval_batch_size = micro_batch_size_per_gpu * calculated_num_gpus
    print(f"计算评估批次大小:")
    print(f"  {micro_batch_size_per_gpu} × {calculated_num_gpus} = {eval_batch_size}")
    
    print(f"\n✅ 结果验证:")
    print(f"  • 推算GPU数量: {calculated_num_gpus}")
    print(f"  • 训练有效批次: {train_batch_size}")
    print(f"  • 评估批次大小: {eval_batch_size}")
    print(f"  • 内存安全: 评估批次({eval_batch_size}) < 训练批次({train_batch_size})")

if __name__ == "__main__":
    explain_batch_size_design()
    verify_calculation() 