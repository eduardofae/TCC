import gc
import torch
from external.BARTScore.bart_score import BARTScorer
from metrics.models import BART_MODEL
from metrics import register_metric

@register_metric
def add_bartscore(prediction, reference, results):
    metric_name = 'bartscore'
    metric = BARTScorer(checkpoint=BART_MODEL)
    precision = metric.score([reference], [prediction])[0]
    recall = metric.score([prediction], [reference])[0]
    results[metric_name] = (precision+recall)/2
    gc.collect()
    torch.cuda.empty_cache()