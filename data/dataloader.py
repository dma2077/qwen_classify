from torch.utils.data import DataLoader
from transformers import AutoProcessor
from dataset import MyFoodDataset
from collate import create_collate_fn

def build_dataloader(
    split_file: str,
    pretrained_model_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
):
    # 1) 准备 processor
    processor = AutoProcessor.from_pretrained(pretrained_model_name)
    # 2) 构造 Dataset
    dataset = MyFoodDataset(split_file)
    # 3) 构造 collate_fn
    collate_fn = create_collate_fn(processor)
    # 4) 用 DataLoader 接管多进程
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader