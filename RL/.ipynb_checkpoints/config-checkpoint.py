from dataclasses import dataclass

@dataclass
class TrainParameters:
    benchmark_name: str = 'squad1'
    sft_policy_ckpt: str = '../SFT_RM_SQUAD1/sft_policy.pth'
    reward_ckpt: str = '../SFT_RM_SQUAD1/outcome_reward.pth'
    tokenizer_ckpt: str = 't5-large'
    num_train_samples: int = 800
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 10
    total_steps: int = 100
    bsz: int = 2
    batch_acc: int = 4
    max_len: int = 48
    min_len: int = 5
    max_eval_len: int = 25
    grad_acc: int = 8
    r_temperature_low: float = 0.75
    r_temperature_high: float = 1
    r_horizon: int = 100
    log_interval: int = 1
    eval_interval: int = 1
    full_eval_interval: int = 100
    seed: int = 0
    buffer_size: int = 4
    logZ_init: float = 1.
    lr_Z: float = 1e-4
    rationale_max_new_tokens: int = 32
    rationale_min_new_tokens: int = 5
    question_max_new_tokens: int = 32
    question_min_new_tokens: int = 5
    rationale_tem_low: float = 0.5
    rationale_tem_high: float = 1
    question_tem_low: float = 0.5
    question_tem_high: float = 1