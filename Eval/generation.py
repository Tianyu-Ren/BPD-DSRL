from datasets import load_dataset
import re
from dataclasses import dataclass
import torch
from tqdm import tqdm
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer
import numpy as np
import random
import argparse


@dataclass
class eval_args:
    q_maxnt: int = 32  # question max length
    q_minnt: int = 5  # question min length
    r_maxnt: int = 32  # rationale max length
    r_minnt: int = 5  # rationale min length
    r_tem: float = 0.75
    q_tem: float = 0.75
    top_p: float = 0.95
    seed: int = 0
    sampling_times: int = 5


def remove_space_before_and_after_punctuation(text):
    corrected_text = re.sub(r'\s([,.?!;:\'"“”‘’])', r'\1', text)
    return corrected_text


def clean_questions(q):
    q = remove_space_before_and_after_punctuation(q)
    if q[-1] != '?':
        q += '?'
    return q


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)


def generate_and_return_rewards(model, encoder_outputs_cache, encoded_answers, eos_token_id, start_token_id,
                                pad_token_id, r_sep_token_id, q_sep_token_id, reward_fn: tuple, max_new_tokens: tuple,
                                min_new_tokens: tuple, temperature: tuple, action_seq=None, skip_rewards=False,
                                top_p=1.):
    batch_size = encoded_answers.size(0)
    encoder_outputs, attention_mask = encoder_outputs_cache
    prompt = torch.cat([encoded_answers.new_ones(batch_size, 1) * start_token_id,
                        encoded_answers], dim=-1)
    decoder_attention_mask = torch.cat([encoded_answers.new_ones(batch_size, 1),
                                        (encoded_answers != pad_token_id) * (encoded_answers != eos_token_id)], dim=-1)
    past_key_values = model(input_ids=None,
                            attention_mask=attention_mask,
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=prompt,
                            decoder_attention_mask=decoder_attention_mask).past_key_values

    rationale_max_new_tokens, rationale_min_new_tokens = max_new_tokens[0], min_new_tokens[0]
    question_max_new_tokens, question_min_new_tokens = max_new_tokens[1], min_new_tokens[1]
    rationale_temperature, question_temperature = temperature[0], temperature[1]
    log_reward_rationale, log_reward_question = None, None
    state = prompt.new_ones(batch_size, 1) * r_sep_token_id
    decoder_attention_mask = torch.cat([decoder_attention_mask, prompt.new_ones(batch_size, 1).bool()], dim=-1)
    if action_seq is not None:
        log_pf = torch.zeros(batch_size, action_seq.size(-1)).float().to(encoder_outputs[0].device)
        for n in range(action_seq.size(-1)):
            outputs = model(input_ids=None,
                            attention_mask=attention_mask,
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=state[:, -1:],
                            decoder_attention_mask=decoder_attention_mask,
                            past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            token_ids = action_seq[:, n]
            state = torch.cat([state, token_ids.unsqueeze(-1)], dim=-1)
            log_pf[:, n] = outputs.logits[:, -1, :].log_softmax(dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            decoder_attention_mask = torch.cat([decoder_attention_mask,
                                                prompt.new_ones(batch_size, 1).bool()], dim=-1)
        log_pf = log_pf * (action_seq != pad_token_id)  # The action_seq is expected to be padded with pad token ids
        rationale_length = ((action_seq == q_sep_token_id).cumsum(dim=-1) < 1).sum(dim=-1) + 1
        rationale_span = torch.arange(action_seq.size(-1),
                                      device=rationale_length.device) <= rationale_length.unsqueeze(-1)
        log_pf_rationale = (log_pf * rationale_span).sum(dim=-1) / rationale_span.sum(dim=-1)
        log_pf_question = (log_pf * ~rationale_span).sum(dim=-1) / ((~rationale_span).sum(dim=-1) -
                                                                    (action_seq == pad_token_id).sum(dim=-1))
        return action_seq, log_pf_rationale, log_pf_question, log_reward_rationale, log_reward_question

    # Generate questions step by step
    # Start sampling rationales
    log_pf_rationale = torch.zeros(batch_size, rationale_max_new_tokens).float().to(encoder_outputs[0].device)
    activate_rationale_seqs = torch.ones(batch_size).bool().to(encoder_outputs[0].device)
    rationale_actions = prompt.new_ones(batch_size, 1) * r_sep_token_id
    for n in range(rationale_max_new_tokens):
        outputs = model(input_ids=None,
                        attention_mask=attention_mask,
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=state[:, -1:],
                        decoder_attention_mask=decoder_attention_mask,
                        past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        with torch.no_grad():
            modified_logits = logits.detach().clone()
            if n < rationale_min_new_tokens:
                modified_logits[:, q_sep_token_id] = -float('inf')
            prob = (modified_logits / rationale_temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, 1)

        token_ids = torch.where(activate_rationale_seqs.unsqueeze(-1),
                                token_ids, torch.ones_like(token_ids) * q_sep_token_id)

        log_prob = logits.log_softmax(dim=-1)
        log_pf_rationale[:, n] = torch.where(activate_rationale_seqs, log_prob.gather(-1, token_ids).squeeze(-1), 0)
        state = torch.cat([state, token_ids], dim=-1)
        rationale_actions = torch.cat([rationale_actions, token_ids], dim=-1)
        # check if all sequences have generated eos
        activate_rationale_seqs = activate_rationale_seqs * (token_ids != q_sep_token_id).squeeze(-1)
        decoder_attention_mask = torch.cat([decoder_attention_mask,
                                            activate_rationale_seqs.unsqueeze(-1)], dim=-1)
        if torch.all(~activate_rationale_seqs):
            break
    if not skip_rewards:
        log_reward_rationale = reward_fn[0](rationale_actions[:, 1:])  #######
    log_pf_rationale = log_pf_rationale.sum(dim=-1) / (rationale_actions != q_sep_token_id).sum(dim=-1)  # -1 + 1
    # Start sampling questions
    state = torch.cat([state[:, :-1], state.new_ones(batch_size, 1) * q_sep_token_id], dim=-1)
    question_actions = prompt.new_ones(batch_size, 1) * q_sep_token_id
    decoder_attention_mask = torch.cat([decoder_attention_mask[:, :-1],
                                        decoder_attention_mask.new_ones(batch_size, 1)], dim=-1)
    log_pf_question = torch.zeros(batch_size, question_max_new_tokens).float().to(encoder_outputs[0].device)
    activate_question_seqs = torch.ones(batch_size).bool().to(encoder_outputs[0].device)
    for n in range(question_max_new_tokens):
        outputs = model(input_ids=None,
                        attention_mask=attention_mask,
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=state[:, -1:],
                        decoder_attention_mask=decoder_attention_mask,
                        past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        with torch.no_grad():
            modified_logits = logits.detach().clone()
            if n < rationale_min_new_tokens:
                modified_logits[:, eos_token_id] = -float('inf')
            prob = (modified_logits / question_temperature).softmax(dim=-1)
            if top_p < 1.:
                # Sort the probabilities and their corresponding indices
                sorted_prob, sorted_indices = torch.sort(prob, descending=True, dim=-1)

                # Compute the cumulative probabilities
                cumulative_prob = torch.cumsum(sorted_prob, dim=-1)

                # Create a mask to filter out the tokens beyond the top-p cumulative probability threshold
                sorted_indices_to_remove = cumulative_prob > top_p

                # Ensure that we always keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Set the probabilities of the tokens beyond the top-p threshold to 0
                sorted_prob[sorted_indices_to_remove] = 0

                # Normalize the new distribution
                top_p_probs = sorted_prob / sorted_prob.sum(dim=-1, keepdim=True)

                # Sample from the top-p distribution
                token_ids = torch.multinomial(top_p_probs, 1).squeeze(1)
                token_ids = sorted_indices[torch.arange(batch_size), token_ids].view(batch_size, 1)
            else:
                token_ids = torch.multinomial(prob, 1)

        token_ids = torch.where(activate_question_seqs.unsqueeze(-1),
                                token_ids, torch.ones_like(token_ids) * eos_token_id)

        log_prob = logits.log_softmax(dim=-1)
        log_pf_question[:, n] = torch.where(activate_question_seqs, log_prob.gather(-1, token_ids).squeeze(-1), 0)
        state = torch.cat([state, token_ids], dim=-1)
        question_actions = torch.cat([question_actions, token_ids], dim=-1)
        # check if all sequences have generated eos
        activate_question_seqs = activate_question_seqs * (token_ids != eos_token_id).squeeze(-1)
        decoder_attention_mask = torch.cat([decoder_attention_mask,
                                            activate_question_seqs.unsqueeze(-1)], dim=-1)
        if torch.all(~activate_question_seqs):
            break
    if not skip_rewards:
        log_reward_question = reward_fn[1](question_actions[:, 1:])  #######
    log_pf_question = log_pf_question.sum(dim=-1) / (question_actions != eos_token_id).sum(dim=-1)  # -1 + 1
    state = clean_state(state[:, 1:], r_sep_token_id, q_sep_token_id, eos_token_id, pad_token_id)
    return state, log_pf_rationale, log_pf_question, log_reward_rationale, log_reward_question


def clean_state(state, r_sep_token_id, q_sep_token_id, eos_token_id, pad_token_id):
    assert torch.all(state[:, 0] != r_sep_token_id)
    new_state = []
    for idx, sample in enumerate(state):
        pos_sep = (sample == q_sep_token_id).nonzero().squeeze(-1)
        rationale, question = sample[:pos_sep[0]], sample[pos_sep[-1]:]
        new_sample = torch.cat([rationale, question], dim=-1)
        new_sample = torch.cat(
            [new_sample[new_sample != eos_token_id], torch.ones(1, device=new_sample.device) * eos_token_id], dim=-1)
        new_state.append(new_sample.int())
    return torch.nn.utils.rnn.pad_sequence(new_state, True, pad_token_id)


def generate(model, tokenizer, eval_batch, eval_config):
    context_ids, answer_ids = eval_batch['input_ids'].cuda(), eval_batch['output_ids'].cuda()
    attention_mask = eval_batch['attention_mask'].cuda()
    encoder_outputs = model.encoder(input_ids=context_ids,
                                    attention_mask=attention_mask)
    encoder_outputs_cache = (encoder_outputs, attention_mask)
    outputs = generate_and_return_rewards(model, encoder_outputs_cache, answer_ids,
                                          tokenizer.eos_token_id,
                                          tokenizer.pad_token_id,
                                          tokenizer.pad_token_id,
                                          tokenizer.vocab['<r>'],
                                          tokenizer.vocab['<q>'],
                                          (None, None),
                                          (eval_config.r_maxnt, eval_config.q_maxnt),
                                          (eval_config.r_minnt, eval_config.q_minnt),
                                          (eval_config.r_tem, eval_config.q_tem),
                                          None,
                                          True,
                                          eval_config.top_p)
    generated_ids, confidence = outputs[0], outputs[2] + outputs[1]
    mask = (generated_ids == tokenizer.vocab['<q>']).cumsum(dim=-1) < 1
    generated_ids[mask] = tokenizer.pad_token_id
    question_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return question_text, confidence.cpu().tolist()


def get_data_loader(dataset, batch_size, sampler, pad_token_id):
    def data_collator(batch):
        input_ids = [torch.tensor(i['input_ids']) for i in batch]
        output_ids = [torch.tensor(i['output_ids']) for i in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = input_ids != pad_token_id
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        question = [i['question'] for i in batch]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'output_ids': output_ids,
                'question': question}

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator)


@torch.inference_mode()
def generate_questions(output_file_name, model_path, n_test_dataset, batch_size, eval_config):

    def load_benchmark(benchmark_name, max_length=384):
        assert benchmark_name in ['squad1', 'squad2', 'newsqa']
        dataset = load_dataset('json', data_files=f'../Datasets/{benchmark_name}/test.json', split='train')

        dataset = dataset.map(lambda x: {'question': clean_questions(x['question'])})

        def tokenize(x):
            input_ids = tokenizer(x['context_hl'], truncation=True, max_length=max_length).input_ids
            output_ids = tokenizer(x['answer']).input_ids
            return {'input_ids': input_ids, 'output_ids': output_ids}

        dataset = dataset.map(tokenize).remove_columns(['context', 'answer'])

        return dataset

    seed_everything(eval_config.seed)
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('t5-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<r>', '<q>', '<hl>']})
    benchmark = load_benchmark(n_test_dataset)
    eval_loader = get_data_loader(benchmark, batch_size, None, tokenizer.pad_token_id)
    results = []
    for batch in tqdm(eval_loader):
        n_batch = []
        confidence_batch = []
        targets = batch['question']
        for num in range(eval_config.sampling_times):
            generated_questions, confidence = generate(model, tokenizer, batch, eval_config)
            n_batch.append(generated_questions)
            confidence_batch.append(confidence)
        batch_n = list(zip(*n_batch))
        batch_n_confidence = list(zip(*confidence_batch))
        for sample_id in range(len(batch_n)):
            sorted_questions = sorted(zip(batch_n[sample_id],
                                          batch_n_confidence[sample_id]), key=lambda x: x[1], reverse=True)
            sorted_questions = list(list(zip(*sorted_questions))[0])
            results.append({'question': sorted_questions, 'target': targets[sample_id]})
    json_results = [json.dumps(i) + '\n' for i in results]
    with open(f'{output_file_name}.jsonl', 'w') as f:
        f.writelines(json_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_maxnt', type=int, default=32, help="Maximum tokens for the question generation")
    parser.add_argument('--q_minnt', type=int, default=5, help="Minimum tokens for the question generation")
    parser.add_argument('--r_maxnt', type=int, default=32, help="Maximum tokens for the rationale generation")
    parser.add_argument('--r_minnt', type=int, default=5, help="Minimum tokens for the rationale generation")
    parser.add_argument('--r_tem', type=float, default=0.75, help="Temperature for the rationale generation")
    parser.add_argument('--q_tem', type=float, default=0.75, help="Temperature for the rationale generation")
    parser.add_argument('--top_p', type=float, default=0.95, help="Nucleus sampling top-p value")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--sampling_times', type=int, default=5, help="Number of times to sample")
    parser.add_argument('--model_path', type=str, help="Model & tokenizer checkpoint path")
    parser.add_argument('--n_test_dataset', type=str,
                        help="test dataset name (i.e., squad1, squad2, newsqa")
    parser.add_argument('--output_file_name', type=str, help="output file saving path")
    parser.add_argument('--eval_bsz', type=int, default=16, help="evaluation batch size")

    args = parser.parse_args()
    generation_args = eval_args()
    generation_args.__dict__.update(
        {key: value for key, value in vars(args).items() if key in vars(generation_args)})

    generate_questions(
        args.output_file_name,
        args.model_path,
        args.n_test_dataset,
        args.eval_bsz,
        generation_args
    )


if __name__ == '__main__':
    main()