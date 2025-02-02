from datasets import load_dataset


def load_rl_train_data(tokenizer, max_length, seed, n_samples=None, n_benchmark='squad1'):

    dataset = load_dataset('json', data_files=f'../Datasets/{n_benchmark}/train.json', split='train')

    if n_samples is not None:
        assert n_samples <= len(dataset)
    else:
        n_samples = len(dataset)

    dataset = dataset.shuffle(seed)[:n_samples]

    if n_benchmark == 'newsqa':
        train_context = dataset['context_truncation']
    else:
        train_context = dataset['context']
    train_rationale = dataset['rationale']
    encoded_train_context = [tokenizer(i, max_length=max_length, truncation=True, return_tensors='pt').input_ids[0]
                             for i in train_context]
    encoded_train_context_hl = [tokenizer(i, max_length=max_length, truncation=True, return_tensors='pt').input_ids[0]
                                for i in dataset['context_hl']]
    train_answer = dataset['answer']
    encoded_train_answer = [tokenizer(i, return_tensors='pt').input_ids[0] for i in train_answer]
    train_question = dataset['question']
    encoded_train_question = [tokenizer(i, return_tensors='pt').input_ids[0] for i in train_question]
    train_attention_mask = [i != tokenizer.pad_token_id for i in encoded_train_context]

    return {
        'train_context': train_context,
        'train_rationale': train_rationale,
        'train_answer': train_answer,
        'encoded_train_context': encoded_train_context,
        'encoded_train_answer': encoded_train_answer,
        'train_question': train_question,
        'encoded_train_question': encoded_train_question,
        'train_attention_mask': train_attention_mask,
        'encoded_train_context_hl': encoded_train_context_hl
    }





