#!/usr/bin/env python3
"""
最简单的WandB eval指标测试
不使用complex define_metric，直接测试基础记录
"""

import wandb
import time

def simple_wandb_test():
    """最简单的eval指标测试"""
    
    run = wandb.init(
        project="simple-eval-test",
        name="basic-eval-test"
    )
    
    print(f"🚀 Run URL: {run.url}")
    
    # 不使用任何define_metric，直接记录数据
    print("📝 直接记录eval数据...")
    
    for step in range(1, 11):
        # 每步记录training数据
        wandb.log({
            "training/loss": 10.0 - step,
            "training/accuracy": step * 0.1,
        }, step=step)
        
        # 每3步记录eval数据
        if step % 3 == 0:
            wandb.log({
                "eval/loss": 8.0 - step * 0.5,
                "eval/accuracy": step * 0.08,
                "eval/samples": 1000,
            }, step=step)
            print(f"Step {step}: 记录了eval数据")
        
        time.sleep(0.5)
    
    print(f"✅ 完成！请检查: {run.url}")
    print("应该看到:")
    print("  - training/loss, training/accuracy 在step 1-10")
    print("  - eval/loss, eval/accuracy, eval/samples 在step 3,6,9")
    
    wandb.finish()

if __name__ == "__main__":
    simple_wandb_test() 