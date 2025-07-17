#!/usr/bin/env python3
"""
简单的wandb测试
"""

import wandb
import time

def test_wandb():
    """测试wandb基本功能"""
    print("🧪 测试wandb基本功能...")
    
    # 初始化wandb
    wandb.init(
        project="simple_test",
        name="basic_test",
        config={"test": True}
    )
    
    print(f"✅ wandb已初始化: {wandb.run.url}")
    
    # 测试基本指标记录
    for i in range(5):
        step = i + 1
        loss = 1.0 - i * 0.1
        acc = 0.5 + i * 0.1
        
        print(f"📊 记录 step={step}: loss={loss}, acc={acc}")
        
        # 记录训练指标
        wandb.log({
            "train/loss": loss,
            "train/accuracy": acc
        }, step=step, commit=True)
        
        # 记录eval指标
        wandb.log({
            "eval/loss": loss + 0.1,
            "eval/accuracy": acc - 0.05
        }, step=step, commit=True)
        
        time.sleep(1)
    
    print("✅ 测试完成！")
    print(f"🔗 请访问: {wandb.run.url}")
    print("💡 检查train和eval指标是否都正确显示和更新")

if __name__ == "__main__":
    test_wandb() 