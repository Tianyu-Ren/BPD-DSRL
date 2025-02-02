#!/bin/bash

model_to_use='t5-large'
task='newsqa'
epoch=5
lr=5e-5
train_bsz_per_device=2
eval_bsz_per_device=32
max_length=384
beta1=0.9
beta2=0.999
warmup_ratio=0.1
num_steps=-1
grad_acc=16
log_interval=10
eval_interval=400
seed=1234
output_dir='warmup_reward'
multi_task_beta=0.5


python3 -m torch.distributed.launch --nproc_per_node=2 reward_main.py --epoch $epoch \
 --train_bsz_per_device $train_bsz_per_device --eval_bsz_per_device $eval_bsz_per_device --max_length $max_length \
 --lr $lr --warmup_ratio $warmup_ratio --grad_acc $grad_acc --model_to_use $model_to_use --beta1 $beta1 --beta2 $beta2 \
  --log_interval $log_interval --eval_interval $eval_interval --seed $seed  --num_steps $num_steps --output_dir $output_dir --multi_task_beta $multi_task_beta --task $task