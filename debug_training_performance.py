import os
import time
import torch
import deepspeed
from tqdm import tqdm

# 设置环境变量
os.environ['NCCL_NTHREADS'] = '64'
os.environ['MASTER_PORT'] = '29501'
os.environ['MASTER_ADDR'] = 'localhost'

def test_training_loop_performance():
    """测试训练循环的纯性能"""
    
    print("🔍 开始性能测试...")
    
    # 初始化分布式
    deepspeed.init_distributed()
    
    # 模拟训练数据
    device = torch.cuda.current_device()
    batch_size = 8
    seq_length = 1024
    
    # 创建模拟数据
    dummy_data = []
    for i in range(100):  # 100个batch
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones((batch_size, seq_length)),
            "pixel_values": torch.randn((batch_size, 3, 224, 224)),
            "labels": torch.randint(0, 10, (batch_size,))
        }
        dummy_data.append(batch)
    
    print(f"📊 准备了 {len(dummy_data)} 个batch，每个batch大小: {batch_size}")
    
    # 测试基本数据传输性能
    print("\n🔥 测试1: 纯数据传输性能")
    start_time = time.time()
    
    for i, batch in enumerate(dummy_data[:20]):
        # 模拟数据传输
        inputs = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        if i % 10 == 0:
            print(f"  处理batch {i+1}/20")
    
    data_transfer_time = time.time() - start_time
    print(f"✅ 数据传输测试完成: {data_transfer_time:.2f}秒 (平均 {data_transfer_time/20:.3f}秒/batch)")
    
    # 测试tqdm性能
    print("\n🔥 测试2: tqdm进度条性能")
    start_time = time.time()
    
    pbar = tqdm(total=20, desc="Testing Progress Bar")
    for i in range(20):
        # 模拟一些工作
        time.sleep(0.01)
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{0.5:.4f}',
            'lr': f'{1e-4:.2e}',
            'step': i+1
        })
    pbar.close()
    
    tqdm_time = time.time() - start_time
    print(f"✅ tqdm测试完成: {tqdm_time:.2f}秒")
    
    # 测试简单的tensor操作
    print("\n🔥 测试3: 基础tensor操作性能")
    start_time = time.time()
    
    for i in range(20):
        # 模拟loss计算
        dummy_loss = torch.tensor(0.5, device=device)
        loss_item = dummy_loss.item()
        
        # 模拟梯度norm计算
        dummy_grad_norm = torch.tensor(1.0, device=device)
        grad_norm_item = dummy_grad_norm.item()
        
        if i % 10 == 0:
            print(f"  处理tensor操作 {i+1}/20")
    
    tensor_ops_time = time.time() - start_time
    print(f"✅ Tensor操作测试完成: {tensor_ops_time:.2f}秒")
    
    # 测试分布式操作（如果有多GPU）
    print("\n🔥 测试4: 分布式操作性能")
    start_time = time.time()
    
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        print(f"  检测到分布式环境: {dist.get_world_size()} GPUs")
        for i in range(10):
            # 测试all_reduce
            test_tensor = torch.tensor(1.0, device=device)
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            result = test_tensor.item()
            
            if i % 5 == 0:
                print(f"  all_reduce操作 {i+1}/10, result: {result}")
    else:
        print("  单GPU环境，跳过分布式测试")
        time.sleep(0.1)  # 模拟一些时间
    
    dist_ops_time = time.time() - start_time
    print(f"✅ 分布式操作测试完成: {dist_ops_time:.2f}秒")
    
    # 总结
    total_time = data_transfer_time + tqdm_time + tensor_ops_time + dist_ops_time
    print(f"\n📊 性能测试总结:")
    print(f"  • 数据传输: {data_transfer_time:.2f}秒")
    print(f"  • tqdm进度条: {tqdm_time:.2f}秒")
    print(f"  • Tensor操作: {tensor_ops_time:.2f}秒")
    print(f"  • 分布式操作: {dist_ops_time:.2f}秒")
    print(f"  • 总时间: {total_time:.2f}秒")
    print(f"  • 平均每个操作: {total_time/20:.3f}秒")

if __name__ == "__main__":
    test_training_loop_performance() 