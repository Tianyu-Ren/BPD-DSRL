import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def prepare_for_question_scoring(question_ids, answer_ids, sep_token_id, eos_token_id, pad_token_id):
    question_answer = []
    sep_token = question_ids.new_ones(1) * sep_token_id
    for question, answer in zip(question_ids, answer_ids):
        answer = answer[answer != pad_token_id]
        if eos_token_id not in question:
            question_answer.append(torch.cat([question, sep_token, answer], dim=-1))
        else:
            eos_pos = ((question == eos_token_id).cumsum(dim=-1) >= 1).nonzero()[0]
            question_answer.append(torch.cat([question[:eos_pos], sep_token, answer], dim=-1))
    return torch.nn.utils.rnn.pad_sequence(question_answer, batch_first=True, padding_value=pad_token_id)


@torch.inference_mode()
def question_score_function(model, encoded_input, encoder_output_cache, sep_token_id, pad_token_id):
    # The encoded_input is the concatenation of question and answer
    encoder_outputs, encoder_attention_mask = encoder_output_cache
    decoder_input_ids = torch.cat([encoded_input.new_ones(encoded_input.size(0), 1) * pad_token_id,
                                   encoded_input[:, :-1]], dim=-1)
    non_pad_token_mask = encoded_input != pad_token_id
    logits = model(input_ids=None,
                   attention_mask=encoder_attention_mask,
                   decoder_input_ids=decoder_input_ids,
                   encoder_outputs=encoder_outputs).logits
    log_prob = logits.log_softmax(dim=-1)
    token_ids = encoded_input  # The target token_ids are exactly the decoder input ids
    log_pf = log_prob.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    log_pf[~non_pad_token_mask] = 0.
    # return log_pf.sum(dim=-1) / (non_pad_token_mask).sum(dim=-1)

    sep_pos = (encoded_input == sep_token_id).nonzero()[:, 1]
    question_span = torch.arange(encoded_input.size(-1), device=encoded_input.device) <= sep_pos.unsqueeze(-1)
    answer_span = ~question_span
    log_pf_question = log_pf * question_span
    log_pf_answer = log_pf * answer_span
    # Reduction is mean
    log_pf_question = log_pf_question.sum(dim=-1) / question_span.sum(dim=-1)
    log_pf_answer = log_pf_answer.sum(dim=-1) / (answer_span.sum(dim=-1) - (~non_pad_token_mask).sum(dim=-1))
    return log_pf_question + log_pf_answer


class QuestionReward:
    def __init__(self, model, sep_token_id, eos_token_id, pad_token_id, temperature=1.):
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.eos_token_id = eos_token_id

    @torch.inference_mode()
    def score(self, question_ids, answer_ids, encoder_outputs_cache):
        encoded_input = prepare_for_question_scoring(question_ids, answer_ids,
                                                     self.sep_token_id, self.eos_token_id, self.pad_token_id)
        log_reward = question_score_function(self.model, encoded_input, encoder_outputs_cache,
                                             self.sep_token_id, self.pad_token_id)
        return log_reward / self.temperature


class RationaleEntailReward:
    def __init__(self, model_path, max_length, temperature, device='cuda'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.temperature = temperature

    @torch.inference_mode()
    def score(self, context_text: list, generated_text: list):
        inputs = self.tokenizer.batch_encode_plus(list(zip(context_text, generated_text)),
                                                  max_length=self.max_length,
                                                  truncation='only_first',
                                                  return_tensors='pt',
                                                  return_token_type_ids=True,
                                                  padding='longest')
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        token_type_ids = inputs['token_type_ids'].to(self.model.device)
        logits = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids).logits
        entailment_score = logits.log_softmax(dim=-1)[:, 0]
        for idx, text in enumerate(generated_text):
            if '?' in text:
                entailment_score[idx] = -10
        return entailment_score / self.temperature


def prepare_for_entailment_scoring(rationale, tokenizer):
    rationale_text = tokenizer.batch_decode(rationale, skip_special_tokens=True)
    rationale_text = [i.replace('<q>', '').strip() for i in rationale_text]
    return rationale_text


def reward_function_question(question_reward_model: QuestionReward, question_ids, answer_ids, encoder_outputs_cache):
    return question_reward_model.score(question_ids, answer_ids, encoder_outputs_cache)


def reward_function_rationale(rationale_reward_model: RationaleEntailReward, rationale_ids, tokenizer, context_text):
    rationale_text = prepare_for_entailment_scoring(rationale_ids, tokenizer)
    return rationale_reward_model.score(context_text, rationale_text)
