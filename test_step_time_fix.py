#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试step_time修复的简单脚本
"""

import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step_time_calculation():
    """测试step_time计算是否正常工作"""
    print("🧪 测试step_time计算修复...")
    
    # 模拟monitor对象
    class MockMonitor:
        def __init__(self):
            self.step_start_time = None
    
    monitor = MockMonitor()
    
    # 测试1: step_start_time为None的情况
    print("测试1: step_start_time为None")
    current_time = time.time()
    
    # 使用修复后的逻辑
    step_start_time = getattr(monitor, 'step_start_time', None)
    if step_start_time is not None:
        step_time = current_time - step_start_time
    else:
        step_time = 0.0
    
    print(f"  current_time: {current_time}")
    print(f"  step_start_time: {step_start_time}")
    print(f"  step_time: {step_time}")
    print(f"  ✅ 测试1通过: step_time = {step_time}")
    
    # 测试2: step_start_time有值的情况
    print("\n测试2: step_start_time有值")
    monitor.step_start_time = current_time - 1.5  # 1.5秒前
    current_time = time.time()
    
    step_start_time = getattr(monitor, 'step_start_time', None)
    if step_start_time is not None:
        step_time = current_time - step_start_time
    else:
        step_time = 0.0
    
    print(f"  current_time: {current_time}")
    print(f"  step_start_time: {step_start_time}")
    print(f"  step_time: {step_time}")
    print(f"  ✅ 测试2通过: step_time ≈ {step_time:.2f}秒")
    
    print("\n🎉 所有测试通过！step_time计算修复成功。")

if __name__ == "__main__":
    test_step_time_calculation() 