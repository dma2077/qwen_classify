import os
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, messages, label = self.data_list[idx]
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
            
            return {
                "image": image,
                "messages": messages,
                "label": label,
            }
        except Exception as e:
            print(f"Error loading image at index {idx}, path {image_path}: {e}")
            print(f"Image exists: {os.path.exists(image_path) if image_path else 'Path is None'}")
            raise
    
class MyFoodDataset(BaseDataset):
    def __init__(self, split_file):
        data_list = []
        import json
        # 每行是 JSON 格式，包含"image_path"和"label"键
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img_path = item.get("image_path")
                label = int(item.get("label"))
                dataset_name = item.get("dataset_name")
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
                data_list.append((img_path, messages, label))
        super().__init__(data_list)
