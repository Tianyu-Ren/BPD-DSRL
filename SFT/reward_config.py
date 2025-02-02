from dataclasses import dataclass


@dataclass
class TrainParams:
    model_to_use: str = 't5-large'
    task: str = 'squad1'
    max_length: int = 384
    lr: float = 5e-5
    train_bsz_per_device: int = 2
    eval_bsz_per_device: int = 32
    grad_acc: int = 16
    beta1: float = 0.9
    beta2: float = 0.999
    epoch: int = 5
    log_interval: int = 10
    eval_interval: int = 400
    output_dir: str = 'reward_output_dir'
    seed: int = 1234
    local_rank: int = -1
    optimizer: str = 'AdamW'
    num_steps: int = -1
    warmup_ratio: float = 0.1
    multi_task_beta: float = 0.5

