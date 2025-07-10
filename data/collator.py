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
        text_list = [
            processor.apply_chat_template(
                messages=m,
                tokenize=False,
                add_generation_prompt=True
            ) for m in msgs
        ]

        # 2) 调用 processor 取得张量
        enc = processor(
            text=text_list,
            images=images,
            return_tensors="pt"
        )
        enc["labels"] = labels
        return enc

    return collate_fn
