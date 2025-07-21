import torch

def create_collate_fn(processor):
    """
    返回 collate_fn：将 batch 中的 PIL.Image + messages 转为模型前向需要的 tensors。
    支持多数据集功能，处理dataset_name和num_classes字段。
    """
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        msgs   = [item["messages"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        
        # 处理数据集名称
        dataset_names = [item.get("dataset_name", "unknown") for item in batch]
        
        # 处理类别数量，如果没有提供则设为None
        num_classes_list = []
        for item in batch:
            num_classes = item.get("num_classes", None)
            num_classes_list.append(num_classes)

        # 1) 先把 messages 转成 chat 文本模板
        text_list = []
        for i, m in enumerate(msgs):
            try:
                text = processor.apply_chat_template(
                    conversation=m,
                    tokenize=False,
                    add_generation_prompt=True
                )
                text_list.append(text)
            except Exception as e:
                print(f"Error processing message {i}: {e}")
                print(f"Message format: {m}")
                raise

        # 2) 调用 processor 取得张量
        try:
            # 先验证图像
            valid_images = []
            for i, img in enumerate(images):
                if img is None:
                    raise ValueError(f"Image {i} is None")
                if not hasattr(img, 'size'):
                    raise ValueError(f"Image {i} is not a valid PIL Image")
                valid_images.append(img)
            
            # 🔥 优化：减少文本长度提升处理速度
            # 对于Qwen2.5-VL，调用processor处理多模态输入
            enc = processor(
                text=text_list,
                images=valid_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024  # 从2048减少到1024提升速度
            )
            
            # 确保所有必要的键都存在
            if "image_grid_thw" not in enc:
                print("Warning: image_grid_thw not found in processor output!")
                print(f"Available keys: {list(enc.keys())}")
                # 如果没有image_grid_thw，尝试生成一个默认值
                # 根据pixel_values的形状推断
                if "pixel_values" in enc and enc["pixel_values"] is not None:
                    batch_size = enc["pixel_values"].shape[0]
                    # 为每个图像创建默认的grid_thw (1, 16, 16) - 这是一个常见的默认值
                    enc["image_grid_thw"] = torch.tensor([[1, 16, 16]] * batch_size, dtype=torch.long)
                    print(f"Generated default image_grid_thw: {enc['image_grid_thw']}")
            
            # 添加标签和多数据集信息
            enc["labels"] = labels
            enc["dataset_names"] = dataset_names
            enc["num_classes_list"] = num_classes_list
            
            return enc
        except Exception as e:
            print(f"Error in processor: {e}")
            print(f"Text list length: {len(text_list)}")
            print(f"Images length: {len(images)}")
            if images:
                print(f"Image sizes: {[getattr(img, 'size', 'NO_SIZE') for img in images]}")
                print(f"Image types: {[type(img) for img in images]}")
            print(f"First few texts: {text_list[:2] if text_list else 'NO_TEXTS'}")
            raise

    return collate_fn
