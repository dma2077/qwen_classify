import torch
import torch.nn.functional as F
from training.losses import *

def test_loss_functions():
    """测试不同损失函数的行为"""
    
    print("🧪 损失函数对比测试")
    print("="*80)
    
    # 模拟数据
    batch_size = 4
    num_classes = 101
    hidden_dim = 768
    
    # 模拟logits和标签
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, hidden_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"测试数据:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Number of classes: {num_classes}")
    print(f"  • Hidden dimension: {hidden_dim}")
    print(f"  • Labels: {labels.tolist()}")
    print()
    
    # 测试不同损失函数
    losses_to_test = [
        ('Cross-Entropy', 'cross_entropy', {}),
        ('Label Smoothing', 'label_smoothing', {'smoothing': 0.1}),
        ('Focal Loss', 'focal', {'gamma': 2.0, 'alpha': 1.0}),
        ('Symmetric CE', 'symmetric_ce', {'alpha': 1.0, 'beta': 1.0}),
        ('ArcFace', 'arcface', {'in_features': hidden_dim, 'out_features': num_classes, 's': 30.0, 'm': 0.5}),
    ]
    
    results = []
    
    for name, loss_type, params in losses_to_test:
        try:
            print(f"🔍 测试 {name}:")
            
            # 创建损失函数
            loss_fn = create_loss_function(loss_type, **params)
            
            # 计算损失
            if loss_type == 'arcface':
                # ArcFace使用原始特征
                loss_value = loss_fn(features, labels)
            else:
                # 其他损失函数使用logits
                loss_value = loss_fn(logits, labels)
            
            results.append((name, loss_value.item()))
            print(f"  ✅ 损失值: {loss_value.item():.4f}")
            print(f"  📋 参数: {params}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        
        print()
    
    # 损失值比较
    print("📊 损失值对比:")
    print("-" * 40)
    for name, value in results:
        print(f"{name:20s}: {value:.4f}")
    
    return results

def analyze_loss_sensitivity():
    """分析损失函数对预测置信度的敏感性"""
    
    print("\n" + "="*80)
    print("📈 损失函数敏感性分析")
    print("="*80)
    
    # 创建不同置信度的预测
    num_classes = 101
    
    # 场景1: 高置信度正确预测
    high_conf_logits = torch.zeros(1, num_classes)
    high_conf_logits[0, 0] = 5.0  # 高置信度预测类别0
    true_label = torch.tensor([0])
    
    # 场景2: 低置信度正确预测
    low_conf_logits = torch.zeros(1, num_classes)
    low_conf_logits[0, 0] = 0.5   # 低置信度预测类别0
    low_conf_logits[0, 1] = 0.3   # 其他类别也有一定概率
    
    # 场景3: 错误预测
    wrong_logits = torch.zeros(1, num_classes)
    wrong_logits[0, 1] = 3.0      # 错误预测类别1
    
    scenarios = [
        ("高置信度正确", high_conf_logits, true_label),
        ("低置信度正确", low_conf_logits, true_label),
        ("高置信度错误", wrong_logits, true_label),
    ]
    
    loss_functions = [
        ('Cross-Entropy', lambda x, y: F.cross_entropy(x, y)),
        ('Focal (γ=2)', lambda x, y: FocalLoss(gamma=2.0)(x, y)),
        ('Label Smoothing', lambda x, y: LabelSmoothingCrossEntropy(smoothing=0.1)(x, y)),
    ]
    
    print(f"{'场景':<15} {'CE Loss':<12} {'Focal Loss':<12} {'Label Smooth':<12}")
    print("-" * 55)
    
    for scenario_name, logits, labels in scenarios:
        losses = []
        for _, loss_fn in loss_functions:
            loss_val = loss_fn(logits, labels).item()
            losses.append(loss_val)
        
        print(f"{scenario_name:<15} {losses[0]:<12.4f} {losses[1]:<12.4f} {losses[2]:<12.4f}")

def print_loss_recommendations():
    """打印损失函数选择建议"""
    
    print("\n" + "="*80)
    print("💡 损失函数选择建议")
    print("="*80)
    
    recommendations = [
        {
            "场景": "类别平衡的食物分类",
            "推荐": "Label Smoothing Cross-Entropy",
            "参数": "smoothing=0.1",
            "优势": "防止过拟合，提高泛化性能，在准确率和鲁棒性间平衡"
        },
        {
            "场景": "类别不平衡的数据集",
            "推荐": "Focal Loss",
            "参数": "gamma=2.0, alpha=1.0",
            "优势": "自动关注难样本，减少易样本的影响，适合长尾分布"
        },
        {
            "场景": "需要高质量特征表示",
            "推荐": "ArcFace Loss",
            "参数": "s=30.0, m=0.5",
            "优势": "提供更好的特征分离度，适合需要细粒度分类的任务"
        },
        {
            "场景": "标签可能有噪声",
            "推荐": "Symmetric Cross-Entropy",
            "参数": "alpha=1.0, beta=1.0",
            "优势": "对标注错误更鲁棒，减少噪声标签的负面影响"
        },
        {
            "场景": "基线比较",
            "推荐": "Standard Cross-Entropy",
            "参数": "无",
            "优势": "简单稳定，计算效率高，适合初步实验"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['场景']}")
        print(f"   📌 推荐: {rec['推荐']}")
        print(f"   ⚙️  参数: {rec['参数']}")
        print(f"   💪 优势: {rec['优势']}")
        print()

def performance_comparison():
    """性能对比提示"""
    
    print("⚡ 性能对比 (计算开销):")
    print("-" * 40)
    print("Cross-Entropy      : ⭐⭐⭐⭐⭐ (最快)")
    print("Label Smoothing    : ⭐⭐⭐⭐  (很快)")
    print("Focal Loss         : ⭐⭐⭐   (中等)")
    print("Symmetric CE       : ⭐⭐    (较慢)")
    print("ArcFace            : ⭐     (最慢)")
    print()
    
    print("🎯 效果提升潜力:")
    print("-" * 40)
    print("Cross-Entropy      : ⭐⭐    (基线)")
    print("Label Smoothing    : ⭐⭐⭐   (稳定提升)")
    print("Focal Loss         : ⭐⭐⭐⭐  (不平衡数据显著)")
    print("ArcFace            : ⭐⭐⭐⭐⭐ (特征质量最佳)")
    print("Symmetric CE       : ⭐⭐⭐   (噪声数据下好)")

if __name__ == "__main__":
    # 运行测试
    test_loss_functions()
    analyze_loss_sensitivity()
    print_loss_recommendations()
    performance_comparison()
    
    print("\n" + "="*80)
    print("🚀 如何在配置中使用:")
    print("="*80)
    print("""
# 在 configs/config.yaml 中配置:

# 1. Label Smoothing (推荐用于平衡数据)
loss:
  type: "label_smoothing"
  smoothing: 0.1
  temperature: 1.0

# 2. Focal Loss (推荐用于不平衡数据)
loss:
  type: "focal"
  alpha: 1.0
  gamma: 2.0

# 3. ArcFace (推荐用于需要高质量特征)
loss:
  type: "arcface"
  s: 30.0
  m: 0.5

# 4. Symmetric CE (推荐用于噪声数据)
loss:
  type: "symmetric_ce"
  alpha: 1.0
  beta: 1.0
    """)
    
    print("📝 训练建议:")
    print("- 从 Label Smoothing 开始，它通常能带来稳定的性能提升")
    print("- 如果数据不平衡，尝试 Focal Loss")
    print("- 如果需要最好的特征表示，使用 ArcFace")
    print("- 可以通过对比实验选择最适合的损失函数")
    print("="*80) 