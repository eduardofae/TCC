from rouge import Rouge
from metrics import register_metric

@register_metric
def add_rouge(prediction, reference, results):
    metric_name = 'rouge'
    metric = Rouge()
    values = metric.get_scores(hyps=prediction, refs=reference)[0]
    results[metric_name+'-1'] = values[metric_name+'-1']['f']
    results[metric_name+'-2'] = values[metric_name+'-2']['f']
    results[metric_name+'-l'] = values[metric_name+'-l']['f']