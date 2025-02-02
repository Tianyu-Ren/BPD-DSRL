from datasets import load_dataset
import torch


def load_data(tokenizer, split='train', max_length=384, task='squad1'):

    data_path = f'../Datasets/{task}/{split}.json'
    dataset = load_dataset('json', data_files=data_path, split='train')

    def tokenize(x):
        context, question, rationale, answer = x['context_hl'], x['question'], x['rationale'], x['answer']
        output_ids = tokenizer(answer + '<r>' + rationale + '<q>' + question).input_ids
        input_ids = tokenizer(context, truncation=True, max_length=max_length).input_ids
        return {'input_ids': input_ids, 'output_ids': output_ids}

    dataset = dataset.map(tokenize)

    return dataset.select_columns(['input_ids', 'output_ids'])


def get_data_loader(dataset, batch_size, sampler, pad_token_id):
    def data_collator(batch):
        input_ids = [torch.tensor(i['input_ids']) for i in batch]
        output_ids = [torch.tensor(i['output_ids']) for i in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = input_ids != pad_token_id
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'output_ids': output_ids}
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator)


