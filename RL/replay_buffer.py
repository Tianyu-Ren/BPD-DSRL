import torch
import heapq
import pickle
import gzip
import numpy as np
import editdistance
from reward_utils import reward_function_question


class ReplayBuffer:

    def __init__(self, max_len, pad_token_id, sim_tolerance=0.25):
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.sim_tolerance = sim_tolerance
        self._buffer = {}

    def reset(self):
        self._buffer = {}

    def add(self, item):
        str_query_answer = item['str_query_answer']
        if item['str_rationale'] in self._buffer[str_query_answer]['exists']:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        tokenized_rationale = [x for x in item['tensor_rationale'].tolist() if x != self.pad_token_id]
        for buffer_item in self._buffer[str_query_answer]['rationales']:
            tokenized_existing_rationale = [x for x in buffer_item[2].tolist() if x != self.pad_token_id]
            if editdistance.eval(tokenized_rationale, tokenized_existing_rationale) < (
                    len(tokenized_rationale) + len(tokenized_existing_rationale)) * self.sim_tolerance:
                if buffer_item[0] >= item['log_rewards']:
                    return
                else:
                    self._buffer[str_query_answer]['exists'].remove(buffer_item[1])
                    self._buffer[str_query_answer]['rationales'].remove(buffer_item)
                    heapq.heapify(self._buffer[str_query_answer]['rationales'])
                    self._buffer[str_query_answer]['exists'].add(item['str_rationale'])
                    heapq.heappush(self._buffer[str_query_answer]['rationales'],
                                   (item['log_rewards'], item['str_rationale'], item['tensor_rationale']))
                    return
        self._buffer[str_query_answer]['exists'].add(item['str_rationale'])
        if len(self._buffer[str_query_answer]['rationales']) >= self.max_len:
            popped = heapq.heappushpop(self._buffer[str_query_answer]['rationales'],
                                       (item['log_rewards'], item['str_rationale'], item['tensor_rationale']))
            self._buffer[str_query_answer]['exists'].remove(popped[1])
        else:
            heapq.heappush(self._buffer[str_query_answer]['rationales'],
                           (
                               item['log_rewards'], item['str_rationale'], item['tensor_rationale']))

    def add_batch(self, query, answer, rationales, log_rewards, tokenizer):
        str_query = ' '.join([str(x) for x in query])
        if answer is not None:
            str_answer = ' '.join([str(x) for x in answer])
        else:
            str_answer = 'None'
        str_query_answer = '|'.join([str_query, str_answer])
        if str_query_answer not in self._buffer:
            self._buffer[str_query_answer] = {'tensor_query': query,
                                              'tensor_answer': answer,
                                              'rationales': [],
                                              'exists': set()}
        token_rationales = tokenizer.batch_decode(rationales)
        for i in range(len(rationales)):
            str_rationale = token_rationales[i].replace(tokenizer.pad_token, '').strip()
            self.add({'log_rewards': log_rewards[i],
                      'str_query_answer': str_query_answer,
                      'str_rationale': str_rationale,
                      'tensor_rationale': rationales[i]})

    def sample(self, batch_size, query, answer):
        str_query = ' '.join([str(x) for x in query])
        if answer is not None:
            str_answer = ' '.join([str(x) for x in answer])
        else:
            str_answer = 'None'
        str_query_answer = '|'.join([str_query, str_answer])
        if str_query_answer not in self._buffer:
            return None, None
        query_answer_buffer = self._buffer[str_query_answer]['rationales']
        if len(query_answer_buffer) < batch_size:
            # if the buffer is not full, use pad_sequence and return all items
            return torch.nn.utils.rnn.pad_sequence([item[2] for item in query_answer_buffer],
                                                   batch_first=True,
                                                   padding_value=self.pad_token_id), \
                torch.tensor([item[0] for item in query_answer_buffer])
        else:
            # do prioritized sampling
            priorities = [item[0] for item in query_answer_buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.min(priorities)  ######
            idx = np.random.choice(len(query_answer_buffer), batch_size,
                                   p=np.exp(priorities) / np.sum(np.exp(priorities)), replace=False)
            return torch.nn.utils.rnn.pad_sequence([query_answer_buffer[i][2] for i in idx],
                                                   batch_first=True,
                                                   padding_value=self.pad_token_id), \
                torch.tensor([query_answer_buffer[i][0] for i in idx])

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]['rationales']:
                print(item[1])
            print('')

    def save(self, path):
        with gzip.open(path, 'wb') as f:
            pickle.dump(self._buffer, f)


@torch.inference_mode()
def load_replay_buffer(buffer_size, load_batch_size, dataset, tokenizer, rationale_reward, question_reward):
    num_samples = len(dataset['train_context'])
    pad_token_id = tokenizer.pad_token_id
    replay_buffer = ReplayBuffer(buffer_size, pad_token_id=pad_token_id)

    def batch_add(lower, upper):
        # Enough for rationale scoring
        text_context_batch = dataset['train_context'][lower:upper]
        text_answer_batch = dataset['train_answer'][lower:upper]
        text_rationale_batch = dataset['train_rationale'][lower:upper]
        log_rewards_rationale = rationale_reward.score(text_context_batch, text_rationale_batch)
        # Enough for question scoring
        encoded_context_batch = dataset['encoded_train_context'][lower:upper]
        encoded_question_batch = dataset['encoded_train_question'][lower:upper]
        encoded_answer_batch = dataset['encoded_train_answer'][lower:upper]
        batched_encoded_context = torch.nn.utils.rnn.pad_sequence(encoded_context_batch, True,
                                                                  padding_value=pad_token_id).cuda()
        attention_mask = batched_encoded_context != pad_token_id
        batched_encoded_question = torch.nn.utils.rnn.pad_sequence(encoded_question_batch, True,
                                                                   padding_value=tokenizer.eos_token_id).cuda()  # !
        batched_encoded_answer = torch.nn.utils.rnn.pad_sequence(encoded_answer_batch, True,
                                                                 padding_value=pad_token_id).cuda()
        reward_encoder_outputs = question_reward.model.encoder(input_ids=batched_encoded_context,
                                                               attention_mask=attention_mask)
        reward_encoder_outputs_cache = (reward_encoder_outputs, attention_mask)

        log_rewards_question = reward_function_question(question_reward, batched_encoded_question,
                                                        batched_encoded_answer, reward_encoder_outputs_cache)
        # concat_rationale_question = []
        encoded_rationale_batch = tokenizer(text_rationale_batch).input_ids
        for i in range(len(encoded_question_batch)):
            rationale_tensor = torch.tensor(encoded_rationale_batch[i][:-1] + [tokenizer.vocab['<q>']])
            concat_rationale_question = torch.cat([rationale_tensor, encoded_question_batch[i]], dim=-1)
            replay_buffer.add_batch(query=text_context_batch[i:i + 1],
                                    answer=text_answer_batch[i:i + 1],
                                    rationales=[concat_rationale_question],
                                    log_rewards=(log_rewards_rationale + log_rewards_question)[i:i + 1].cpu(),
                                    tokenizer=tokenizer)

    start = 0
    end = start + load_batch_size
    while end < num_samples:
        batch_add(start, end)
        start += load_batch_size
        end = start + load_batch_size
    batch_add(start, num_samples)
    return replay_buffer
