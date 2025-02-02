import torch


def generate_and_return_rewards(model, encoder_outputs_cache, encoded_answers, eos_token_id, start_token_id,
                                pad_token_id, r_sep_token_id, q_sep_token_id, reward_fn: tuple, max_new_tokens: tuple,
                                min_new_tokens: tuple, temperature: tuple, action_seq=None, skip_rewards=False):

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
        rationale_span = torch.arange(action_seq.size(-1), device=rationale_length.device) <= rationale_length.unsqueeze(-1)
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
        new_sample = torch.cat([new_sample[new_sample != eos_token_id], torch.ones(1, device=new_sample.device) * eos_token_id], dim=-1)
        new_state.append(new_sample.int())
    return torch.nn.utils.rnn.pad_sequence(new_state, True, pad_token_id)

