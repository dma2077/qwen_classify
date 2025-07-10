import os
import yaml
import torch
import sys
from transformers import TrainingArguments, Trainer

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data.dataloader import build_dataloader
from training.model import load_config, build_model
from optimizer.optimizer import build_optimizer
from training.lr_scheduler import build_scheduler

class DataLoaderTrainer(Trainer):
    """
    Subclass of HuggingFace Trainer that uses external DataLoaders
    instead of Dataset objects.
    """
    def __init__(
        self,
        train_loader,
        eval_loader,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def get_train_dataloader(self):
        return self.train_loader

    def get_eval_dataloader(self):
        return self.eval_loader


def main():
    # 1. Load config
    cfg = load_config()

    # 2. Build DataLoaders
    train_loader = build_dataloader(
        split_file=cfg["data"]["train_jsonl"],
        pretrained_model_name=cfg["model"]["pretrained_name"],
        batch_size=cfg["training"]["micro_batch_size_per_gpu"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
    )
    eval_loader = build_dataloader(
        split_file=cfg["data"]["val_jsonl"],
        pretrained_model_name=cfg["model"]["pretrained_name"],
        batch_size=cfg["training"]["micro_batch_size_per_gpu"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=False,
    )

    # 3. Build model, optimizer, scheduler
    model = build_model(cfg).cuda()
    optimizer = build_optimizer(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    num_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = build_scheduler(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=num_steps,
    )

    # 4. Prepare TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["micro_batch_size_per_gpu"],
        per_device_eval_batch_size=cfg["training"]["micro_batch_size_per_gpu"],
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=cfg["training"].get("logging_steps", 50),
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=cfg.get("accelerate", {}).get("fp16", False),
        dataloader_drop_last=True,  # Important for gradient accumulation
        push_to_hub=False,
    )

    # 5. Instantiate trainer with DataLoaders
    trainer = DataLoaderTrainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, scheduler),
        train_loader=train_loader,
        eval_loader=eval_loader,
        data_collator=None,  # already batched by DataLoader
    )

    # 6. Launch training
    trainer.train()


if __name__ == "__main__":
    main()
