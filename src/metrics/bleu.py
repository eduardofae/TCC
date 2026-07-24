import evaluate
from math import log, exp
from metrics import register_metric

def calc_n_bleu(precisions, bp):
    sum = 0
    for p in precisions:
        if p == 0: return 0
        sum += log(p)
    sum /= len(precisions)
    return bp * exp(sum)

@register_metric
def add_bleu(prediction, reference, results):
    metric_name = 'bleu'
    metric = evaluate.load('bleu')
    values = metric.compute(predictions=[prediction], references=[reference])
    results[metric_name] = values[metric_name]
    results[metric_name+'-1'] = calc_n_bleu(values['precisions'][:1], values['brevity_penalty'])
    results[metric_name+'-2'] = calc_n_bleu(values['precisions'][:2], values['brevity_penalty'])