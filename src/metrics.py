import evaluate
from rouge import Rouge 
from external.moverscore.moverscore_v2 import word_mover_score
from collections import defaultdict
from external.BARTScore.bart_score import BARTScorer
from math import log, exp
from tqdm import tqdm

def calc_n_bleu(precisions, bp):
    sum = 0
    for p in precisions:
        if p == 0: return 0
        sum += log(p)
    sum /= len(precisions)
    return bp * exp(sum)

metrics = {
    'bleu': evaluate.load('bleu'),
    'meteor': evaluate.load('meteor'),
    'rouge': Rouge(),
    'bertscore': evaluate.load('bertscore'),
    'bartscore': BARTScorer(checkpoint='adalbertojunior/bart-base-portuguese')
}

def add_bleu(prediction, reference, results):
    metric_name = 'bleu'
    metric = metrics[metric_name]
    values = metric.compute(predictions=[prediction], references=[reference])
    results[metric_name] = values[metric_name]
    results[metric_name+'-1'] = calc_n_bleu(values['precisions'][:1], values['brevity_penalty'])
    results[metric_name+'-2'] = calc_n_bleu(values['precisions'][:2], values['brevity_penalty'])

def add_meteor(prediction, reference, results):
    metric_name = 'meteor'
    metric = metrics[metric_name]
    values = metric.compute(predictions=[prediction], references=[reference])
    results[metric_name] = values[metric_name]

def add_rouge(prediction, reference, results):
    metric_name = 'rouge'
    metric = metrics[metric_name]
    values = metric.get_scores(hyps=prediction, refs=reference)[0]
    results[metric_name+'-1'] = values[metric_name+'-1']['f']
    results[metric_name+'-2'] = values[metric_name+'-2']['f']
    results[metric_name+'-l'] = values[metric_name+'-l']['f']

def add_bertscore(prediction, reference, results):
    metric_name = 'bertscore'
    metric = metrics[metric_name]
    values = metric.compute(predictions=[prediction], references=[reference], model_type='neuralmind/bert-large-portuguese-cased', num_layers=24)
    results[metric_name+'-p']  = values['precision'][0]
    results[metric_name+'-r']  = values['recall'][0]
    results[metric_name+'-f1'] = values['f1'][0]

def add_moverscore(prediction, reference, results):
    metric_name = 'moverscore'
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    values = word_mover_score([reference], [prediction], idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    results[metric_name] = values[0]

def add_bartscore(prediction, reference, results):
    metric_name = 'bartscore'
    metric = metrics[metric_name]
    precision = metric.score([reference], [prediction])[0]
    recall = metric.score([prediction], [reference])[0]
    results[metric_name] = (precision+recall)/2

def add_metrics(prediction, reference, task):
    results = {}
    add_bleu(prediction, reference, results)
    add_rouge(prediction, reference, results)
    add_meteor(prediction, reference, results)
    add_bertscore(prediction, reference, results)
    add_moverscore(prediction, reference, results)
    add_bartscore(prediction, reference, results)
    task['metrics'] = results

def eval_data(data, data_type):
    results = data.copy()
    print(f'Starting {data_type} Evaluations!')
    for task in tqdm(results.values()):
        add_metrics(task['generated'], task['reference'], task)
    return results