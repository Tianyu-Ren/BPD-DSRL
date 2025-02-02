import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup
from reward_utils import QuestionReward, RationaleEntailReward, reward_function_question, reward_function_rationale
from policy_utils import generate_and_return_rewards
from load_rl_data import load_rl_train_data
from replay_buffer import load_replay_buffer
from Z_model import LogZ
from config import TrainParameters
from tqdm import tqdm
import random
import numpy as np


def DSRL(config: TrainParameters, policy, tokenizer, logz, question_reward_model,
         rationale_reward_model, replay_buffer, dataset):
    policy_max_new_tokens = (config.rationale_max_new_tokens, config.question_max_new_tokens)
    policy_min_new_tokens = (config.rationale_min_new_tokens, config.question_min_new_tokens)
    rationale_tem_span = (config.rationale_tem_low, config.rationale_tem_high)
    question_tem_span = (config.question_tem_low, config.question_tem_high)

    optimizer = torch.optim.AdamW([{'params': policy.parameters(), 'lr': config.lr},
                                   {'params': logz.parameters(), 'lr': config.lr_Z}])

    lr_schedular = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=config.warmup_steps,
                                                   num_training_steps=config.total_steps)

    rt_schedular = lambda x: (config.r_temperature_high - (config.r_temperature_high -
                                                           config.r_temperature_low) * min(1, x / config.r_horizon))

    batch_size, grad_acc = config.bsz, config.grad_acc
    batch_acc = config.batch_acc

    encoded_train_context = dataset['encoded_train_context']
    encoded_train_context_hl = dataset['encoded_train_context_hl']
    encoded_train_answer = dataset['encoded_train_answer']
    train_attention_mask = dataset["train_attention_mask"]
    scalar = torch.cuda.amp.GradScaler()

    for step in tqdm(range(config.total_steps)):
        question_reward_model.temperature = rt_schedular(step)
        rationale_reward_model.temperature = rt_schedular(step)
        step_loss = 0.
        for n in range(grad_acc):
            policy_choice = random.choice([0, 1, 2])
            sample_id = np.random.choice(np.arange(len(encoded_train_context)))
            context = encoded_train_context[sample_id].cuda()
            context_hl = encoded_train_context_hl[sample_id].cuda()
            context_text = dataset['train_context'][sample_id]
            answer = encoded_train_answer[sample_id].cuda()
            answer_text = dataset['train_context'][sample_id]
            attention_mask = train_attention_mask[sample_id].cuda()
            if policy_choice in [0, 1]:
                context = context.repeat(batch_size, 1)
                context_hl = context_hl.repeat(batch_size, 1)
                answer = answer.repeat(batch_size, 1)
                context_text = [context_text] * batch_size
                attention_mask = attention_mask.repeat(batch_size, 1)
                attention_mask_hl = context_hl != tokenizer.pad_token_id
                if policy_choice == 0:
                    policy_temperature = (1., 1.)
                else:
                    rationale_tem = (random.random() *
                                     (rationale_tem_span[1] - rationale_tem_span[0]) + rationale_tem_span[0])
                    question_tem = (random.random() *
                                    (question_tem_span[1] - question_tem_span[0]) + question_tem_span[0])
                    policy_temperature = (rationale_tem, question_tem)
                encoder_outputs = policy.encoder(input_ids=context_hl,
                                                 attention_mask=attention_mask_hl)
                with torch.no_grad():
                    encoder_outputs_reward = question_reward_model.model.encoder(input_ids=context,
                                                                                 attention_mask=attention_mask)
                encoder_outputs_cache = (encoder_outputs, attention_mask_hl)
                encoder_outputs_cache_reward = (encoder_outputs_reward, attention_mask)
                log_Z_value = logz(input_ids=context_hl,
                                   attention_mask=attention_mask_hl)
                for bsz_acc in range(batch_acc):
                    generated_text, log_pf_rationale, log_pf_question, log_reward_rationale, log_reward_question = (
                        generate_and_return_rewards(
                            policy,
                            encoder_outputs_cache,
                            answer,
                            tokenizer.eos_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.vocab['<r>'],
                            tokenizer.vocab['<q>'],
                            (lambda x: reward_function_rationale(rationale_reward_model,
                                                                 x,
                                                                 tokenizer,
                                                                 context_text),
                             lambda x: reward_function_question(question_reward_model,
                                                                x,
                                                                answer,
                                                                encoder_outputs_cache_reward)),
                            policy_max_new_tokens,
                            policy_min_new_tokens,
                            policy_temperature,
                            None,
                            False
                        ))
                    log_pf = log_pf_rationale + log_pf_question

                    log_rewards = log_reward_rationale + log_reward_question
                    # log_rewards = log_reward_question ######################
                    # log_rewards = log_reward_rationale

                    loss = (log_pf + log_Z_value - log_rewards) ** 2
                    loss = (loss.mean() / grad_acc / batch_acc)
                    scalar.scale(loss).backward(retain_graph=True)
                    step_loss += loss.item()

                    raw_log_rewards = (log_reward_rationale * rationale_reward_model.temperature +
                                       log_reward_question * question_reward_model.temperature)

                    replay_buffer.add_batch(query=context_text[:1],
                                            answer=[answer_text],
                                            rationales=generated_text,
                                            log_rewards=raw_log_rewards,
                                            tokenizer=tokenizer)
            else:  # off policy
                action_seq, log_rewards = replay_buffer.sample(batch_size, query=context, answer=answer)
                if action_seq is None:
                    continue
                context_hl = context_hl.repeat(action_seq.size(0), 1)
                attention_mask_hl = context_hl != tokenizer.pad_token_id
                answer = answer.repeat(action_seq.size(0), 1)
                encoder_outputs = policy.encoder(input_ids=context_hl, attention_mask=attention_mask_hl)
                encoder_outputs_cache = (encoder_outputs, attention_mask_hl)
                log_Z_value = logz(input_ids=context_hl,
                                   attention_mask=attention_mask_hl)
                log_rewards *= (1 / rationale_reward_model.temperature)  # redo the effect of reward tempering

                generated_text, log_pf_rationale, log_pf_question, _, __ = (
                    generate_and_return_rewards(
                        policy,
                        encoder_outputs_cache,
                        answer,
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.vocab['<r>'],
                        tokenizer.vocab['<q>'],
                        (None, None),
                        policy_max_new_tokens,
                        policy_min_new_tokens,
                        (1., 1.),
                        action_seq,
                        True
                    ))
                log_pf = log_pf_rationale + log_pf_question
                loss = (log_pf + log_Z_value - log_rewards) ** 2
                loss = (loss.mean() / grad_acc)
                loss.backward()
                step_loss += loss.item()
        scalar.step(optimizer)
        scalar.update()
        lr_schedular.step()
        optimizer.zero_grad()
        if step % config.log_interval == 0:
            print(f'loss: {step_loss}')
        if step % config.eval_interval == 0:
            sample_id = np.random.choice(np.arange(len(encoded_train_context)))
            context = encoded_train_context[sample_id].cuda().repeat(batch_size, 1)
            context_hl = encoded_train_context_hl[sample_id].cuda().repeat(batch_size, 1)
            attention_mask_hl = context_hl != tokenizer.pad_token_id
            context_text = [dataset['train_context'][sample_id]] * batch_size
            answer = encoded_train_answer[sample_id].cuda().repeat(batch_size, 1)
            attention_mask = train_attention_mask[sample_id].cuda().repeat(batch_size, 1)
            with torch.no_grad():
                encoder_outputs = policy.encoder(input_ids=context_hl, attention_mask=attention_mask_hl)
                encoder_outputs_reward = question_reward_model.model.encoder(input_ids=context,
                                                                             attention_mask=attention_mask)

                encoder_outputs_cache = (encoder_outputs, attention_mask_hl)
                encoder_outputs_cache_reward = (encoder_outputs_reward, attention_mask)

                generated_text, log_pf_rationale, log_pf_question, log_reward_rationale, log_reward_question = (
                    generate_and_return_rewards(
                        policy,
                        encoder_outputs_cache,
                        answer,
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.vocab['<r>'],
                        tokenizer.vocab['<q>'],
                        (lambda x: reward_function_rationale(rationale_reward_model,
                                                             x,
                                                             tokenizer,
                                                             context_text),
                         lambda x: reward_function_question(question_reward_model,
                                                            x,
                                                            answer,
                                                            encoder_outputs_cache_reward)),
                        policy_max_new_tokens,
                        policy_min_new_tokens,
                        (1., 1.),
                        None,
                        False
                    ))
            # print(log_rewards)
            print(f'Reward: 'f'\n{log_reward_rationale + log_reward_question}')
            print(f'Generated questions: \n{tokenizer.batch_decode(generated_text)}')
            print(f'Context: \n{dataset["train_context"][sample_id]}')
            print(f'Gold question: \n{dataset["train_question"][sample_id]}')
            print(f'Gold Answer:\n{dataset["train_answer"][sample_id]}')
        if (step + 1) % 100 == 0:
            policy.save_pretrained(f'DSRL-checkpoint-{config.benchmark_name}-step-{step+1}')


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    train_args = TrainParameters()

    seed_everything(train_args.seed)

    policy = T5ForConditionalGeneration.from_pretrained(train_args.sft_policy_ckpt).cuda()

    log_z_model = LogZ(train_args.sft_policy_ckpt).cuda()

    q_reward = T5ForConditionalGeneration.from_pretrained(train_args.reward_ckpt).cuda()

    tokenizer = AutoTokenizer.from_pretrained(train_args.tokenizer_ckpt)

    tokenizer.add_special_tokens({'additional_special_tokens': ['<r>', '<q>', '<hl>']})

    q_reward_model = QuestionReward(q_reward,
                                    tokenizer.vocab['<hl>'],
                                    tokenizer.eos_token_id,
                                    tokenizer.pad_token_id)
    
    r_reward_model = RationaleEntailReward('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                                           max_length=256,
                                           temperature=1.)

    dataset = load_rl_train_data(tokenizer,
                                 max_length=384,
                                 seed=train_args.seed,
                                 n_samples=train_args.num_train_samples,
                                 n_benchmark=train_args.benchmark_name)

    replay_buffer = load_replay_buffer(train_args.buffer_size,
                                       32,
                                       dataset,
                                       tokenizer,
                                       r_reward_model,
                                       q_reward_model)

    DSRL(train_args, policy, tokenizer, log_z_model, q_reward_model, r_reward_model, replay_buffer, dataset)

main()
