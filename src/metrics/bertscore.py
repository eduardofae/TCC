import gc
import torch
import evaluate
from metrics.models import BERT_MODEL
from metrics import register_metric

@register_metric
def add_bertscore(prediction, reference, results):
    metric_name = 'bertscore'
    metric = evaluate.load(metric_name)
    values = metric.compute(predictions=[prediction], references=[reference], model_type=BERT_MODEL, num_layers=24)
    results[metric_name+'-p']  = values['precision'][0]
    results[metric_name+'-r']  = values['recall'][0]
    results[metric_name+'-f1'] = values['f1'][0]
    gc.collect()
    torch.cuda.empty_cache()