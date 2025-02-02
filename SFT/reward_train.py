from policy_train import PolicyTrainer, seed_everything
from tqdm import tqdm
import torch
import math
from transformers import get_linear_schedule_with_warmup


class RewardTrainer(PolicyTrainer):
    def __init__(self, model, tokenizer, train_set, args, eval_set=None):
        super().__init__(model, tokenizer, train_set, args, eval_set)
        self.task_type = 'reward'
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.multi_task_beta = args.multi_task_beta

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

    def compute_loss(self, batch):
        logits, target_ids = self.tf_predict(batch)
        labels = target_ids.clone().cuda()
        labels[labels == self.tokenizer.pad_token_id] = -100
        # Treat it as multi-task learning

        question_length = ((labels == self.tokenizer.vocab['<hl>']).cumsum(dim=-1) < 1).sum(dim=-1) + 1
        question_mask = torch.arange(labels.size(-1), device=labels.device) <= question_length.view(-1, 1)
        question_mask = question_mask.view(-1)
        answer_mask = ~question_mask
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(logits, labels)
        valid_q_tokens = (labels[question_mask] != -100).int().sum(dim=-1)
        valid_a_tokens = (labels[answer_mask] != -100).int().sum(dim=-1)
        loss_qg = loss[question_mask].sum() / valid_q_tokens
        loss_qa = loss[answer_mask].sum() / valid_a_tokens
        return loss_qg, loss_qa

    def compute_metrics(self, batch):
        return self.compute_loss(batch)

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
                loss_qg, loss_qa = self.compute_loss(batch)
                loss = self.multi_task_beta * loss_qg + (1 - self.multi_task_beta) * loss_qa

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
                            total_qa_perplexity = 0.
                            total_qg_perplexity = 0.
                            with torch.no_grad():
                                for eval_batch in tqdm(self.eval_iter, desc='Evaluating', total=len(self.eval_iter)):
                                    qg_perplexity, qa_perplexity = self.compute_metrics(eval_batch)
                                    total_qg_perplexity += qg_perplexity
                                    total_qa_perplexity += qa_perplexity
                            mean_perplexity = 0.5 * (total_qa_perplexity + total_qg_perplexity) / len(self.eval_iter)
                            mean_qg_perplexity = total_qg_perplexity / len(self.eval_iter)
                            mean_qa_perplexity = total_qa_perplexity / len(self.eval_iter)

                            print(f"Epoch: {epoch}, Step: {update_step}, Score: {mean_perplexity:.4f}, QG Score: "
                                  f"{mean_qg_perplexity}, QA Score: {mean_qa_perplexity}")

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
