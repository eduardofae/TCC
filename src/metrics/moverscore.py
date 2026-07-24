import os
import gc
import torch
from collections import defaultdict
from metrics.models import BERT_MODEL
from metrics import register_metric

os.environ['MOVERSCORE_MODEL'] = BERT_MODEL
from external.moverscore.moverscore_v2 import word_mover_score

@register_metric
def add_moverscore(prediction, reference, results):
    metric_name = 'moverscore'
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    values = word_mover_score([reference], [prediction], idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    results[metric_name] = values[0]
    gc.collect()
    torch.cuda.empty_cache()