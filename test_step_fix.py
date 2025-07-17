#!/usr/bin/env python3
"""
测试step修复 - 验证effective_step和global_step的使用
"""

import os
import time
import json
import torch
import wandb
from typing import Dict

# 模拟配置
config = {
    'output_dir': './test_output',
    'wandb': {
        'enabled': True,
        'project': 'qwen_classify_test',
        'run_name': 'step_fix_test',
        'tags': ['test', 'step_fix']
    },
    'monitor': {
        'freq': {
            'all_freq': 1  # 每步都记录
        }
    },
    'model': {
        'max_sequence_length': 512
    },
    'deepspeed': {
        'train_batch_size': 32
    }
}

class StepTestMonitor:
    """测试step使用的监控器"""
    
    def __init__(self, output_dir: str, config: Dict):
        self.output_dir = output_dir
        self.config = config
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        # 初始化wandb
        self._init_wandb()
        
    def _init_wandb(self):
        """初始化wandb"""
        try:
            wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb']['run_name'],
                tags=self.config['wandb']['tags'],
                config=self.config
            )
            
            # 定义所有指标组
            wandb.define_metric("step")
            wandb.define_metric("training/*", step_metric="step")
            wandb.define_metric("perf/*", step_metric="step")
            wandb.define_metric("eval/*", step_metric="step")
            
            print("✅ WandB初始化成功")
            print(f"📊 项目: {wandb.run.project}")
            print(f"🔗 运行: {wandb.run.name}")
            print(f"🚀 查看地址: {wandb.run.url}")
            
        except Exception as e:
            print(f"❌ WandB初始化失败: {e}")
            return
    
    def log_step(self, step: int, epoch: int, loss: float, grad_norm: float, learning_rate: float):
        """记录训练步骤"""
        current_time = time.time()
        step_time = current_time - self.step_start_time
        
        # 准备指标数据
        wandb_data = {
            "step": int(step),
            "training/loss": float(loss),
            "training/lr": float(learning_rate),
            "training/epoch": float(epoch),
            "training/grad_norm": float(grad_norm),
            "perf/step_time": float(step_time),
            "perf/steps_per_second": float(1.0 / step_time) if step_time > 0 else 0.0,
        }
        
        # 记录到wandb
        try:
            wandb.log(wandb_data, step=int(step), commit=True)
            print(f"✅ Step {step}: 已记录训练指标")
        except Exception as e:
            print(f"❌ 记录训练指标失败: {e}")
        
        self.step_start_time = current_time
    
    def log_evaluation(self, step: int, eval_loss: float, eval_accuracy: float):
        """记录评估指标"""
        eval_data = {
            "step": int(step),
            "eval/overall_loss": float(eval_loss),
            "eval/overall_accuracy": float(eval_accuracy),
        }
        
        try:
            wandb.log(eval_data, step=int(step), commit=True)
            print(f"✅ Eval Step {step}: 已记录评估指标")
        except Exception as e:
            print(f"❌ 记录评估指标失败: {e}")
    
    def finish(self):
        """结束训练"""
        if wandb.run is not None:
            total_time = time.time() - self.start_time
            wandb.log({"training/total_time": total_time}, commit=True)
            wandb.finish()
            print(f"📊 训练完成，总耗时: {total_time:.2f}秒")

def simulate_training_with_gradient_accumulation():
    """模拟带梯度累积的训练过程"""
    print("🚀 开始模拟带梯度累积的训练")
    print("=" * 60)
    
    # 创建监控器
    monitor = StepTestMonitor('./test_output', config)
    
    # 模拟参数
    gradient_accumulation_steps = 4  # 梯度累积步数
    total_effective_steps = 20       # 总有效步数
    eval_steps = 5                   # 评估间隔
    
    print(f"📈 训练参数:")
    print(f"   • 梯度累积步数: {gradient_accumulation_steps}")
    print(f"   • 总有效步数: {total_effective_steps}")
    print(f"   • 评估间隔: {eval_steps}")
    print(f"   • 总global步数: {total_effective_steps * gradient_accumulation_steps}")
    print("=" * 60)
    
    # 模拟训练循环
    global_step = 0
    effective_step = 0
    
    for epoch in range(2):  # 2个epoch
        print(f"📊 Epoch {epoch + 1}/2")
        
        for batch_idx in range(total_effective_steps * gradient_accumulation_steps):
            global_step += 1
            
            # 模拟训练数据
            loss = 2.0 - (effective_step * 0.05) + (torch.rand(1).item() * 0.1)
            grad_norm = 0.5 + (torch.rand(1).item() * 0.3)
            learning_rate = 1e-5 * (0.95 ** (effective_step // 10))
            
            # 检查是否是有效步骤（完成了梯度累积）
            is_effective_step = global_step % gradient_accumulation_steps == 0
            
            if is_effective_step:
                effective_step += 1
                
                print(f"   Global Step {global_step} -> Effective Step {effective_step}")
                
                # 记录训练步骤（使用effective_step）
                monitor.log_step(effective_step, epoch, loss, grad_norm, learning_rate)
                
                # 每eval_steps步进行一次评估
                if effective_step % eval_steps == 0:
                    # 模拟评估数据
                    eval_loss = loss * 0.8 + (torch.rand(1).item() * 0.1)
                    eval_accuracy = 0.3 + (effective_step * 0.02) + (torch.rand(1).item() * 0.05)
                    eval_accuracy = min(eval_accuracy, 0.95)
                    
                    # 记录评估指标（使用effective_step）
                    monitor.log_evaluation(effective_step, eval_loss, eval_accuracy)
                    print(f"     📊 评估: loss={eval_loss:.4f}, acc={eval_accuracy:.4f}")
            
            # 短暂暂停
            time.sleep(0.05)
    
    # 结束训练
    monitor.finish()
    
    print("=" * 60)
    print("✅ 测试完成！")
    print("📊 请在WandB界面中检查：")
    print("   • 是否还有'Steps must be monotonically increasing'警告")
    print("   • 所有指标是否都使用统一的step轴")
    print("   • training、perf、eval指标是否都正常显示")

if __name__ == "__main__":
    simulate_training_with_gradient_accumulation() 