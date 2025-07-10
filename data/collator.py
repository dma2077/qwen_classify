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
        enc = processor(
            text=text_list,
            images=images,
            return_tensors="pt"
        )
        enc["labels"] = labels
        return enc

    return collate_fn
