import torch

def create_collate_fn(processor):
    """
    返回 collate_fn：将 batch 中的 PIL.Image + messages 转为模型前向需要的 tensors。
    """
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        msgs   = [item["messages"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

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
            
            # 对于Qwen2.5-VL，调用processor处理多模态输入
            enc = processor(
                text=text_list,
                images=valid_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # 限制文本长度
            )
            enc["labels"] = labels
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
