import os.path
import torch
import math
from transformers import get_linear_schedule_with_warmup
from policy_data import get_data_loader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PolicyTrainer:
    def __init__(self, model, tokenizer, train_set, args, eval_set=None):
        self.task_type = 'policy'
        self.args = args
        self.opt = None
        self.model = model
        self.lr = args.lr
        self.grad_acc = args.grad_acc
        self.epoch = args.epoch
        self.linear_schedular = None
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.tokenizer = tokenizer
        if torch.cuda.device_count() > 1:
            self.distributed = True
            train_sampler = DistributedSampler(train_set)
        else:
            self.distributed = False
            train_sampler = None
        self.train_sampler = train_sampler
        self.train_iter = get_data_loader(train_set, args.train_bsz_per_device, train_sampler, tokenizer.pad_token_id)
        if eval_set is not None:
            self.eval_iter = get_data_loader(eval_set, args.eval_bsz_per_device, None, tokenizer.pad_token_id)
        else:
            self.eval_iter = None
        self.output_dir = self.args.output_dir
        self.total_update_steps = self.__get_total_steps()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __create_optimizer(self):

        if self.args.optimizer == 'AdamW':
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(self.args.beta1, self.args.beta2))
        elif self.args.optimizer == 'Adam':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.args.beta1, self.args.beta2))
        else:
            raise NotImplementedError

    def __get_total_steps(self):
        if self.args.num_steps != -1:
            return self.args.num_steps
        update_steps_per_epoch = math.ceil(len(self.train_iter) / self.grad_acc)
        return update_steps_per_epoch * self.epoch

    def __create_linear_schedular(self):
        total_steps = self.__get_total_steps()
        warmup_steps = int(self.args.warmup_ratio * total_steps)
        self.linear_schedular = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=warmup_steps,
                                                                num_training_steps=total_steps)

    def tf_predict(self, batch):
        encoder_input_ids, target_ids = batch['input_ids'].cuda(), batch['output_ids']
        encoder_attention_mask = batch['attention_mask'].cuda()
        if self.distributed:
            decoder_input_ids = self.model.module.prepare_decoder_input_ids_from_labels(target_ids).cuda()
        else:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(target_ids).cuda()
        logits = self.model(encoder_input_ids,
                            attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input_ids).logits
        return logits, target_ids

    def compute_loss(self, batch):
        logits, target_ids = self.tf_predict(batch)
        labels = target_ids.clone().cuda()
        labels[labels == self.tokenizer.pad_token_id] = -100
        answer_length = ((labels == self.tokenizer.vocab['<r>']).cumsum(dim=-1) < 1).sum(dim=-1) + 1
        answer_mask = torch.arange(labels.size(-1), device=labels.device) <= answer_length.view(-1, 1)
        labels[answer_mask] = -100
        return self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    @torch.inference_mode()
    def compute_metrics(self, batch):
        validation_loss = self.compute_loss(batch)
        return torch.exp(validation_loss)  # Turn into perplexity

    def train(self):

        seed_everything(self.args.seed)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.args.local_rank],
                                                                   output_device=self.args.local_rank, )
        else:
            self.args.local_rank = 0
        self.model.train()

        self.__create_optimizer()

        self.__create_linear_schedular()

        scalar = torch.cuda.amp.GradScaler()

        best_metric = 99999.

        for epoch in tqdm(range(self.epoch), desc='EPOCH', total=self.epoch, disable=self.args.local_rank == 1):

            update_step = 0

            self.train_sampler.set_epoch(epoch) if self.train_sampler is not None else print("Training on a single GPU")

            grad_acc_loss = 0.

            for batch_idx, batch in tqdm(enumerate(self.train_iter), desc='Training', total=len(self.train_iter),
                                         disable=self.args.local_rank == 1):
                loss = self.compute_loss(batch)  # ********
                loss = loss / self.grad_acc
                grad_acc_loss += loss.item()
                scalar.scale(loss).backward()

                if ((batch_idx + 1) % self.grad_acc == 0) or (batch_idx + 1 == len(self.train_iter)):

                    scalar.step(self.opt)
                    scalar.update()
                    self.linear_schedular.step()
                    update_step += 1
                    self.opt.zero_grad()

                    if update_step % self.args.log_interval == 0 and self.args.local_rank == 0:
                        print(f"Epoch: {epoch}, Step: {update_step}, Loss: {grad_acc_loss}")
                    grad_acc_loss = 0.

                    if ((update_step % self.args.eval_interval == 0 or update_step == self.total_update_steps)
                            and self.eval_iter is not None):
                        if self.args.local_rank == 0:
                            self.model.eval()
                            total_perplexity = 0.
                            with torch.no_grad():
                                for eval_batch in tqdm(self.eval_iter, desc='Evaluating', total=len(self.eval_iter)):
                                    perplexity = self.compute_metrics(eval_batch)
                                    total_perplexity += perplexity
                            mean_perplexity = total_perplexity / len(self.eval_iter)
                            print(f"Epoch: {epoch}, Step: {update_step}, Score: {mean_perplexity:.4f}")
                            if mean_perplexity < best_metric:
                                best_metric = mean_perplexity
                                if self.distributed:
                                    self.model.module.save_pretrained(
                                        f'{self.output_dir}/ckpt-{self.args.model_to_use}-best-{self.task_type}-{self.args.task}.pth')
                                else:
                                    self.model.save_pretrained(
                                        f'{self.output_dir}/ckpt-{self.args.model_to_use}-best-{self.task_type}-{self.args.task}.pth')
                            self.model.train()
                        torch.distributed.barrier()
        if self.args.local_rank == 0:
            if self.distributed:
                self.model.module.save_pretrained(
                    f'{self.output_dir}/ckpt-{self.args.model_to_use}-final-{self.task_type}-{self.args.task}.pth')
            else:
                self.model.save_pretrained(
                    f'{self.output_dir}/checkpoint-{self.args.model_to_use}-final-{self.task_type}-{self.args.task}.pth')
