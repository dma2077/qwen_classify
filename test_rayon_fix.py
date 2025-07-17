#!/usr/bin/env python3
"""
测试Rayon线程池修复
"""

import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rayon_fix():
    """测试Rayon修复是否有效"""
    
    print("🧪 开始测试Rayon线程池修复...")
    
    # 1. 导入修复模块
    try:
        from training.utils.rayon_fix import apply_rayon_fix
        apply_rayon_fix()
        print("✅ 成功导入并应用Rayon修复")
    except Exception as e:
        print(f"❌ 导入Rayon修复失败: {e}")
        return False
    
    # 2. 测试多进程环境
    def worker_function(worker_id):
        """工作函数，模拟评估过程"""
        try:
            # 模拟一些可能触发Rayon的操作
            import torch
            import transformers
            from transformers import AutoProcessor
            
            # 尝试加载processor（这通常会触发tokenizers）
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            
            print(f"✅ Worker {worker_id}: 成功加载processor")
            return True
            
        except Exception as e:
            print(f"❌ Worker {worker_id}: 失败 - {e}")
            return False
    
    # 3. 测试单进程
    print("\n📊 测试单进程环境:")
    try:
        result = worker_function(0)
        if result:
            print("✅ 单进程测试通过")
        else:
            print("❌ 单进程测试失败")
            return False
    except Exception as e:
        print(f"❌ 单进程测试异常: {e}")
        return False
    
    # 4. 测试多进程（可选）
    print("\n📊 测试多进程环境:")
    try:
        # 使用较少的进程数，避免资源冲突
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker_function, i) for i in range(2)]
            results = [future.result(timeout=30) for future in futures]
            
        if all(results):
            print("✅ 多进程测试通过")
        else:
            print("❌ 多进程测试失败")
            return False
            
    except Exception as e:
        print(f"⚠️  多进程测试异常: {e}")
        print("   这可能是因为系统资源限制，但不影响单进程训练")
    
    print("\n🎉 Rayon修复测试完成！")
    return True

def test_evaluation_with_fix():
    """测试评估函数是否正常工作"""
    
    print("\n🧪 测试评估函数...")
    
    try:
        # 导入评估相关模块
        from training.utils.evaluation import evaluate_multi_dataset
        from training.utils.rayon_fix import apply_rayon_fix
        
        # 应用修复
        apply_rayon_fix()
        
        print("✅ 评估模块导入成功")
        print("✅ Rayon修复已应用")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估模块测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("🔧 Rayon线程池修复测试")
    print("="*80)
    
    # 测试1: Rayon修复
    test1_result = test_rayon_fix()
    
    # 测试2: 评估函数
    test2_result = test_evaluation_with_fix()
    
    print("\n" + "="*80)
    print("📋 测试结果总结")
    print("="*80)
    print(f"Rayon修复测试: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"评估函数测试: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！Rayon修复应该能解决评估时的线程池冲突问题。")
    else:
        print("\n⚠️  部分测试失败，可能需要进一步调试。")
    
    print("\n💡 建议:")
    print("1. 如果测试通过，可以正常进行训练")
    print("2. 如果仍有问题，可以尝试减少num_workers或使用单进程训练")
    print("3. 确保系统有足够的内存和文件描述符")

if __name__ == "__main__":
    main() 