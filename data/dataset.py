import os
import random
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_data = self.data_list[idx]
        # 支持新格式：(image_path, messages, label, dataset_name, num_classes)
        if len(item_data) == 5:
            image_path, messages, label, dataset_name, num_classes = item_data
        else:
            # 兼容旧格式：(image_path, messages, label)
            image_path, messages, label = item_data
            dataset_name = "unknown"
            num_classes = None
            
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 加载并验证图像
            image = Image.open(image_path).convert("RGB")
            
            # 确保图像不为空且有合理的尺寸
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image size: {image.size}")
            
            # 检查图像模式
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            result = {
                "image": image,
                "messages": messages,
                "label": label,
                "dataset_name": dataset_name,
            }
            
            # 只有在有效时才添加num_classes
            if num_classes is not None:
                result["num_classes"] = num_classes
            
            return result
        except Exception as e:
            print(f"Error loading image at index {idx}, path {image_path}: {e}")
            print(f"Image exists: {os.path.exists(image_path) if image_path else 'Path is None'}")
            raise

class MultiDatasetLoader(BaseDataset):
    """
    多数据集加载器，支持从多个文件读取数据，shuffle，以及部分评估
    """
    def __init__(self, jsonl_file_list, dataset_configs=None, shuffle_datasets=True, 
                 eval_ratios=None, is_eval=False, use_partial_eval=True):
        """
        Args:
            jsonl_file_list: 数据文件路径列表
            dataset_configs: 数据集配置字典
            shuffle_datasets: 是否shuffle所有数据
            eval_ratios: 各数据集的评估比例字典 {dataset_name: ratio}
            is_eval: 是否为评估模式
            use_partial_eval: 评估时是否使用部分数据
        """
        self.dataset_configs = dataset_configs or {}
        self.shuffle_datasets = shuffle_datasets
        self.eval_ratios = eval_ratios or {}
        self.is_eval = is_eval
        self.use_partial_eval = use_partial_eval
        self._original_file_list = jsonl_file_list  # 保存原始文件列表
        
        # 收集所有数据
        all_data = []
        dataset_stats = {}
        
        for jsonl_file in jsonl_file_list:
            if not os.path.exists(jsonl_file):
                print(f"⚠️ 跳过不存在的文件: {jsonl_file}")
                continue
                
            dataset_data = self._load_single_file(jsonl_file)
            all_data.extend(dataset_data)
            
            # 统计各数据集的数据量
            for _, _, _, dataset_name, _ in dataset_data:
                if dataset_name not in dataset_stats:
                    dataset_stats[dataset_name] = 0
                dataset_stats[dataset_name] += 1
        
        # 打印数据集统计信息
        print(f"\n📊 多数据集加载统计 ({'评估模式' if is_eval else '训练模式'}):")
        total_samples = len(all_data)
        for dataset_name, count in dataset_stats.items():
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            eval_ratio = self.eval_ratios.get(dataset_name, 1.0)
            if is_eval and use_partial_eval:
                actual_count = int(count * eval_ratio)
                print(f"  • {dataset_name}: {count:,} → {actual_count:,} ({eval_ratio:.1%}) samples ({percentage:.1f}%)")
            else:
                print(f"  • {dataset_name}: {count:,} samples ({percentage:.1f}%)")
        print(f"📊 总计: {total_samples:,} samples")
        
        # 如果是评估模式且使用部分评估，按比例采样
        if is_eval and use_partial_eval:
            all_data = self._apply_eval_sampling(all_data)
            print(f"📊 部分评估后总计: {len(all_data):,} samples")
        
        # Shuffle数据
        if shuffle_datasets:
            random.shuffle(all_data)
            print("🔀 数据已shuffle")
        
        super().__init__(all_data)
    
    def _load_single_file(self, jsonl_file):
        """加载单个jsonl文件"""
        import json
        data_list = []
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if not item:  # 跳过空行
                        continue
                        
                    img_path = item.get("image_path")
                    label = int(item.get("label"))
                    dataset_name = item.get("dataset_name", "unknown")
                    
                    # 从配置中获取该数据集的类别数
                    dataset_config = self.dataset_configs.get(dataset_name, {})
                    num_classes = dataset_config.get("num_classes", None)
                    
                    # 构造 HF chat-format messages (Qwen2.5-VL格式)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": f"This is an image of {dataset_name}, what dish is it?"},
                            ],
                        }
                    ]
                    data_list.append((img_path, messages, label, dataset_name, num_classes))
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析错误 {jsonl_file}:{line_num}: {e}")
                except Exception as e:
                    print(f"⚠️ 数据处理错误 {jsonl_file}:{line_num}: {e}")
        
        return data_list
    
    def _apply_eval_sampling(self, all_data):
        """对评估数据按数据集比例采样"""
        # 按数据集分组
        dataset_groups = {}
        for item in all_data:
            dataset_name = item[3]  # dataset_name位置
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(item)
        
        # 按比例采样
        sampled_data = []
        for dataset_name, items in dataset_groups.items():
            eval_ratio = self.eval_ratios.get(dataset_name, 1.0)
            sample_count = int(len(items) * eval_ratio)
            
            if sample_count < len(items):
                # 随机采样
                sampled_items = random.sample(items, sample_count)
            else:
                sampled_items = items
            
            sampled_data.extend(sampled_items)
        
        return sampled_data
    
    def get_full_dataset(self):
        """返回完整数据集（不进行部分采样）"""
        return MultiDatasetLoader(
            jsonl_file_list=self._original_file_list,
            dataset_configs=self.dataset_configs,
            shuffle_datasets=False,  # 完整评估时不shuffle
            eval_ratios=self.eval_ratios,
            is_eval=True,
            use_partial_eval=False  # 不使用部分评估
        )

class MyFoodDataset(BaseDataset):
    def __init__(self, split_file, dataset_configs=None):
        """
        Args:
            split_file: jsonl文件路径
            dataset_configs: 数据集配置字典，格式为 {dataset_name: {"num_classes": int, ...}}
        """
        self.dataset_configs = dataset_configs or {}
        data_list = []
        import json
        # 每行是 JSON 格式，包含"image_path"、"label"和可选的"dataset_name"键
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img_path = item.get("image_path")
                label = int(item.get("label"))
                dataset_name = item.get("dataset_name", "unknown")
                
                # 从配置中获取该数据集的类别数
                dataset_config = self.dataset_configs.get(dataset_name, {})
                num_classes = dataset_config.get("num_classes", None)
                
                # 构造 HF chat-format messages (Qwen2.5-VL格式)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": f"This is an image of {dataset_name}, what dish is it?"},
                        ],
                    }
                ]
                data_list.append((img_path, messages, label, dataset_name, num_classes))
        super().__init__(data_list)
