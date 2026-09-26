import torch


def get_warmup_cosine(optimizer, steps_per_epoch, epochs):
    warmup_ratio = 0.1

    total_steps = steps_per_epoch * epochs

    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(1, warmup_steps)
    warmup_steps = min(warmup_steps, total_steps - 1)

    cosine_steps = total_steps - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0.0)

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )