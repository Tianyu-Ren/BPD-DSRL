from sacrebleu import sentence_bleu, corpus_bleu
import numpy as np
import random
import json
import argparse
from tqdm import tqdm


def calculate_self_bleu(predictions):
    sent_bleu_list = []
    for i in range(len(predictions)):
        h = predictions[i]
        ref = predictions[:i] + predictions[i + 1:]
        sent_bleu_list.append(sentence_bleu(h, ref).score)
    return np.mean(sent_bleu_list)


def get_prediction_with_highest_confidence(predictions):
    return predictions[0]  # We have sorted the predictions with their confidence


def get_prediction_with_highest_sentence_bleu(predictions, target):
    sentence_bleu_hyp = [sentence_bleu(prediction, [target]).score for prediction in predictions]
    if np.all((np.array(sentence_bleu_hyp) < 0.0001)):
        best_hyp = random.choice(predictions)
    else:
        best_hyp = predictions[sentence_bleu_hyp.index(max(sentence_bleu_hyp))]
    return best_hyp


def evaluation(file_path):
    with open(file_path, 'r') as f:
        result = f.readlines()
    result_dict = [json.loads(i) for i in result]
    top_1_predictions = [get_prediction_with_highest_confidence(i['question']) for i in result_dict]
    oracle_predictions = [get_prediction_with_highest_sentence_bleu(i['question'], i['target'])
                          for i in tqdm(result_dict)]
    self_bleu_scores = [calculate_self_bleu(i['question']) for i in tqdm(result_dict)]
    top_1_accuracy = corpus_bleu(top_1_predictions, [[i['target'] for i in result_dict]]).score
    oracle_accuracy = corpus_bleu(oracle_predictions, [[i['target'] for i in result_dict]]).score
    return top_1_accuracy, oracle_accuracy, np.mean(self_bleu_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_path', type=str)
    args = parser.parse_args()
    top_1, oracle, self_ = evaluation(args.output_file_path)
    print(f'Top-1 Metric: {top_1}; Oracle Metric: {oracle}; Self Metric: {self_}')

if __name__ == '__main__':
    main()
    # dataset_names = ['SQUAD1', 'SQUAD2', 'NEWSQA']  # SQUAD 1 & 2 correspond to SQUAD 1.1 / 1 and SQUAD 1.1 / 2
    # for name in dataset_names:
    #     path = f'Test_Results_BPD_DSRL/{name}-BPD-DSRL.jsonl'
    #     top_1, oracle, self_ = evaluation(path)
    #     print(f'Top-1 Metric on {name}: {top_1}; Oracle Metric on {name}: {oracle}; Self Metric on {name}: {self_}')