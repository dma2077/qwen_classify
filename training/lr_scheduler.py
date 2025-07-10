from transformers import get_cosine_schedule_with_warmup

def build_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
