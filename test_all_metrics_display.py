#!/usr/bin/env python3
"""
完整指标显示测试 - 确保training、perf、eval指标都能正常显示
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
        'run_name': 'all_metrics_test',
        'tags': ['test', 'all_metrics']
    },
    'monitor': {
        'freq': {
            'all_freq': 1  # 每步都记录所有指标，确保能看到
        }
    },
    'model': {
        'max_sequence_length': 512
    },
    'deepspeed': {
        'train_batch_size': 32
    }
}

class SimpleTrainingMonitor:
    """简化的训练监控器，确保所有指标都能显示"""
    
    def __init__(self, output_dir: str, config: Dict):
        self.output_dir = output_dir
        self.config = config
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        # 初始化wandb
        self._init_wandb()
        
        # 设置监控频率 - 每步都记录
        self.freq = {
            'training_log_freq': 1,
            'perf_log_freq': 1,
            'gpu_log_freq': 1,
        }
        
        # 模拟模型参数
        self.batch_size = 32
        self.seq_length = 512
        
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
        """记录训练步骤 - 每步都记录所有指标"""
        current_time = time.time()
        step_time = current_time - self.step_start_time
        
        # 准备所有指标数据
        wandb_data = {
            # 统一的step字段
            "step": int(step),
            
            # Training组指标
            "training/loss": float(loss),
            "training/lr": float(learning_rate),
            "training/epoch": float(epoch),
            "training/grad_norm": float(grad_norm),
            
            # Perf组指标
            "perf/step_time": float(step_time),
            "perf/steps_per_second": float(1.0 / step_time) if step_time > 0 else 0.0,
            "perf/tokens_per_second": float(self.batch_size * self.seq_length / step_time) if step_time > 0 else 0.0,
            "perf/samples_per_second": float(self.batch_size / step_time) if step_time > 0 else 0.0,
        }
        
        # 添加GPU指标（如果可用）
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                memory_utilization = (memory_allocated / memory_total) * 100
                
                wandb_data.update({
                    "perf/gpu_memory_allocated_gb": float(memory_allocated),
                    "perf/gpu_memory_reserved_gb": float(memory_reserved),
                    "perf/gpu_memory_utilization_percent": float(memory_utilization),
                })
            except Exception as e:
                print(f"⚠️ GPU指标获取失败: {e}")
        
        # 记录到wandb
        try:
            wandb.log(wandb_data, step=int(step), commit=True)
            print(f"✅ Step {step}: 已记录 {len(wandb_data)} 个指标")
            print(f"   Training: {[k for k in wandb_data.keys() if k.startswith('training/')]}")
            print(f"   Perf: {[k for k in wandb_data.keys() if k.startswith('perf/')]}")
        except Exception as e:
            print(f"❌ 记录指标失败: {e}")
        
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
            print(f"   Eval: {list(eval_data.keys())}")
        except Exception as e:
            print(f"❌ 记录评估指标失败: {e}")
    
    def finish(self):
        """结束训练"""
        if wandb.run is not None:
            total_time = time.time() - self.start_time
            wandb.log({"training/total_time": total_time}, commit=True)
            wandb.finish()
            print(f"📊 训练完成，总耗时: {total_time:.2f}秒")

def main():
    """主测试函数"""
    print("🚀 开始完整指标显示测试")
    print("=" * 60)
    
    # 创建监控器
    monitor = SimpleTrainingMonitor('./test_output', config)
    
    # 模拟训练过程
    total_steps = 20
    eval_steps = 5
    
    print(f"📈 开始训练，总步数: {total_steps}, 评估间隔: {eval_steps}")
    print("=" * 60)
    
    for step in range(1, total_steps + 1):
        # 模拟训练数据
        epoch = step / 10.0
        loss = 2.0 - (step * 0.05) + (torch.rand(1).item() * 0.1)  # 递减的损失
        grad_norm = 0.5 + (torch.rand(1).item() * 0.3)  # 随机梯度范数
        learning_rate = 1e-5 * (0.95 ** (step // 10))  # 递减的学习率
        
        # 记录训练步骤
        monitor.log_step(step, epoch, loss, grad_norm, learning_rate)
        
        # 每eval_steps步进行一次评估
        if step % eval_steps == 0:
            # 模拟评估数据
            eval_loss = loss * 0.8 + (torch.rand(1).item() * 0.1)  # 评估损失略低于训练损失
            eval_accuracy = 0.3 + (step * 0.02) + (torch.rand(1).item() * 0.05)  # 递增的准确率
            eval_accuracy = min(eval_accuracy, 0.95)  # 限制在95%以内
            
            monitor.log_evaluation(step, eval_loss, eval_accuracy)
        
        # 短暂暂停，模拟实际训练
        time.sleep(0.1)
    
    # 结束训练
    monitor.finish()
    
    print("=" * 60)
    print("✅ 测试完成！")
    print("📊 请在WandB界面中查看以下指标组：")
    print("   • training/* - 训练指标")
    print("   • perf/* - 性能指标") 
    print("   • eval/* - 评估指标")
    print("   • system/* - 系统指标（自动生成）")
    print("🔗 如果指标没有立即显示，请刷新页面或等待几秒钟")

if __name__ == "__main__":
    main() 