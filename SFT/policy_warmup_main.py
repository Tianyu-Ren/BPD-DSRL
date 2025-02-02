from policy_train import PolicyTrainer
from policy_config import TrainParams
from policy_data import load_data
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import argparse


def main():
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_to_use', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_bsz_per_device', type=int)
    parser.add_argument('--eval_bsz_per_device', type=int)
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--warmup_ratio', type=float)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--grad_acc', type=int)
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--log_interval', type=int)
    parser.add_argument('--eval_interval', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    train_args = TrainParams()
    train_var = {i: j for i, j in args.__dict__.items() if i in train_args.__dict__}
    train_args = TrainParams(**train_var)
    torch.cuda.set_device(args.local_rank)
    print(train_args)
    model = T5ForConditionalGeneration.from_pretrained(train_args.model_to_use)
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_to_use)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<r>', '<q>', '<hl>']})
    train = load_data(tokenizer, max_length=train_args.max_length, split='train', task=train_args.task)
    dev = load_data(tokenizer, max_length=train_args.max_length, split='dev', task=train_args.task)
    model.resize_token_embeddings(len(tokenizer))
    trainer = PolicyTrainer(
        model,
        tokenizer,
        train,
        train_args,
        dev
    )
    trainer.train()


if __name__ == '__main__':
    main()
