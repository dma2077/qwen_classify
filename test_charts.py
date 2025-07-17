#!/usr/bin/env python3
"""
测试wandb图表功能
模拟训练和评估数据，验证loss和accuracy随step的变化规律
"""

import time
import random
import wandb
import numpy as np

def test_wandb_charts():
    """测试wandb图表功能"""
    
    # 初始化wandb
    wandb.init(
        project="qwen-classify-test",
        name="chart-test",
        config={
            "model": "qwen2.5-vl-7b",
            "learning_rate": 1e-5,
            "batch_size": 32
        }
    )
    
    print("🚀 开始测试wandb图表功能...")
    print(f"🔗 访问链接: {wandb.run.url}")
    
    # 定义指标
    wandb.define_metric("training/loss", summary="min")
    wandb.define_metric("training/accuracy", summary="max")
    wandb.define_metric("eval/loss", summary="min")
    wandb.define_metric("eval/accuracy", summary="max")
    
    # 模拟训练数据
    steps = 100
    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []
    
    # 生成模拟数据（下降的loss，上升的accuracy）
    for step in range(steps):
        # 训练指标：loss下降，accuracy上升
        train_loss = 2.0 * np.exp(-step / 30) + 0.1 + random.uniform(-0.05, 0.05)
        train_acc = 0.1 + 0.8 * (1 - np.exp(-step / 25)) + random.uniform(-0.02, 0.02)
        
        # 评估指标：类似趋势但更平滑
        eval_loss = 2.2 * np.exp(-step / 35) + 0.15 + random.uniform(-0.03, 0.03)
        eval_acc = 0.08 + 0.75 * (1 - np.exp(-step / 30)) + random.uniform(-0.01, 0.01)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_acc)
        
        # 记录训练指标（每个step）
        wandb.log({
            "training/loss": train_loss,
            "training/accuracy": train_acc,
            "global_step": step
        }, commit=True)
        
        # 记录评估指标（每10个step）
        if step % 10 == 0:
            wandb.log({
                "eval/loss": eval_loss,
                "eval/accuracy": eval_acc,
                "global_step": step
            }, commit=True)
            
            print(f"Step {step:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")
        
        time.sleep(0.1)  # 模拟训练时间
    
    # 创建自定义图表
    print("📊 创建自定义图表...")
    
    # 训练vs评估对比图表
    wandb.log({
        "charts/training_vs_eval": wandb.plot.line_series(
            xs=[list(range(steps))],
            ys=[train_losses, eval_losses, train_accuracies, eval_accuracies],
            keys=["Train Loss", "Eval Loss", "Train Acc", "Eval Acc"],
            title="Training vs Evaluation Metrics Over Time",
            xname="Step"
        )
    }, commit=True)
    
    # 学习率变化图表
    lr_steps = list(range(0, steps, 10))
    lr_values = [1e-5 * (0.9 ** (step // 20)) for step in lr_steps]
    
    wandb.log({
        "charts/learning_rate": wandb.plot.line_series(
            xs=[lr_steps],
            ys=[lr_values],
            keys=["Learning Rate"],
            title="Learning Rate Schedule",
            xname="Step"
        )
    }, commit=True)
    
    # 梯度范数图表
    grad_norms = [1.0 * np.exp(-step / 40) + 0.1 + random.uniform(-0.02, 0.02) for step in range(steps)]
    
    wandb.log({
        "charts/gradient_norm": wandb.plot.line_series(
            xs=[list(range(steps))],
            ys=[grad_norms],
            keys=["Gradient Norm"],
            title="Gradient Norm Over Time",
            xname="Step"
        )
    }, commit=True)
    
    print("✅ 图表创建完成！")
    print(f"🔗 请访问wandb界面查看图表: {wandb.run.url}")
    
    # 显示最终统计
    print("\n📈 最终统计:")
    print(f"训练Loss: {min(train_losses):.4f} -> {train_losses[-1]:.4f}")
    print(f"训练Accuracy: {train_accuracies[0]:.4f} -> {train_accuracies[-1]:.4f}")
    print(f"评估Loss: {min(eval_losses):.4f} -> {eval_losses[-1]:.4f}")
    print(f"评估Accuracy: {eval_accuracies[0]:.4f} -> {eval_accuracies[-1]:.4f}")
    
    wandb.finish()
    print("🎉 测试完成！")

if __name__ == "__main__":
    test_wandb_charts() 