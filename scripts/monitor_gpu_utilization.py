#!/usr/bin/env python3
"""
GPU利用率实时监控脚本
用于监控训练过程中的GPU性能瓶颈
"""

import time
import json
import subprocess
import argparse
from datetime import datetime

def get_gpu_stats():
    """获取GPU状态信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [x.strip() for x in line.split(',')]
                gpu_stats.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'gpu_util': int(parts[2]),
                    'mem_util': int(parts[3]),
                    'mem_used': int(parts[4]),
                    'mem_total': int(parts[5]),
                    'power_draw': float(parts[6]) if parts[6] != '[N/A]' else 0,
                    'power_limit': float(parts[7]) if parts[7] != '[N/A]' else 0,
                    'temperature': int(parts[8])
                })
        return gpu_stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def analyze_performance(gpu_stats):
    """分析性能瓶颈"""
    analysis = {
        'low_gpu_util': [],
        'high_mem_usage': [],
        'thermal_throttling': [],
        'power_limiting': [],
        'recommendations': []
    }
    
    for gpu in gpu_stats:
        gpu_id = gpu['index']
        
        # GPU利用率低
        if gpu['gpu_util'] < 85:
            analysis['low_gpu_util'].append(gpu_id)
        
        # 内存使用率高
        if gpu['mem_util'] > 95:
            analysis['high_mem_usage'].append(gpu_id)
        
        # 温度过高
        if gpu['temperature'] > 80:
            analysis['thermal_throttling'].append(gpu_id)
        
        # 功耗限制
        if gpu['power_draw'] > gpu['power_limit'] * 0.95 and gpu['power_limit'] > 0:
            analysis['power_limiting'].append(gpu_id)
    
    # 生成建议
    if analysis['low_gpu_util']:
        analysis['recommendations'].extend([
            "🔧 GPU利用率低，建议：",
            "  • 增加batch size (train_micro_batch_size_per_gpu)",
            "  • 增加数据加载线程 (num_workers)",
            "  • 减少保存/日志频率",
            "  • 检查数据预处理耗时"
        ])
    
    if analysis['high_mem_usage']:
        analysis['recommendations'].extend([
            "⚠️  内存使用率高，建议：",
            "  • 启用gradient checkpointing",
            "  • 考虑使用ZeRO Stage 3",
            "  • 减少batch size"
        ])
    
    if analysis['thermal_throttling']:
        analysis['recommendations'].extend([
            "🌡️  温度过高，建议：",
            "  • 检查散热系统",
            "  • 降低batch size",
            "  • 增加风扇转速"
        ])
    
    return analysis

def print_gpu_status(gpu_stats):
    """打印GPU状态"""
    print(f"\n{'='*80}")
    print(f"GPU状态监控 - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    for gpu in gpu_stats:
        mem_pct = (gpu['mem_used'] / gpu['mem_total']) * 100
        power_pct = (gpu['power_draw'] / gpu['power_limit']) * 100 if gpu['power_limit'] > 0 else 0
        
        # 利用率颜色编码
        util_color = ""
        if gpu['gpu_util'] >= 90:
            util_color = "🟢"  # 绿色 - 优秀
        elif gpu['gpu_util'] >= 70:
            util_color = "🟡"  # 黄色 - 一般
        else:
            util_color = "🔴"  # 红色 - 差
        
        print(f"GPU {gpu['index']} {util_color} | "
              f"Util: {gpu['gpu_util']:3d}% | "
              f"Mem: {mem_pct:5.1f}% ({gpu['mem_used']:5d}/{gpu['mem_total']:5d}MB) | "
              f"Temp: {gpu['temperature']:2d}°C | "
              f"Power: {power_pct:5.1f}% ({gpu['power_draw']:6.1f}W)")

def main():
    parser = argparse.ArgumentParser(description='GPU utilization monitor')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--log-file', type=str, help='Log file to save statistics')
    parser.add_argument('--analyze', action='store_true', help='Enable performance analysis')
    args = parser.parse_args()
    
    log_data = []
    
    try:
        while True:
            gpu_stats = get_gpu_stats()
            if not gpu_stats:
                print("No GPU data available")
                time.sleep(args.interval)
                continue
            
            # 打印状态
            print_gpu_status(gpu_stats)
            
            # 性能分析
            if args.analyze:
                analysis = analyze_performance(gpu_stats)
                
                if analysis['recommendations']:
                    print(f"\n📊 性能分析:")
                    for rec in analysis['recommendations']:
                        print(rec)
            
            # 记录日志
            if args.log_file:
                timestamp = datetime.now().isoformat()
                log_entry = {
                    'timestamp': timestamp,
                    'gpu_stats': gpu_stats
                }
                log_data.append(log_entry)
                
                # 每分钟保存一次日志
                if len(log_data) % (60 // args.interval) == 0:
                    with open(args.log_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
        if args.log_file and log_data:
            with open(args.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"日志已保存到: {args.log_file}")

if __name__ == "__main__":
    main() 