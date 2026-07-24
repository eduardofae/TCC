from tqdm import tqdm
from autoregistry import Registry
from utils.importer import import_all_submodules

register_metric = Registry()
import_all_submodules(__name__)

def add_metrics(prediction, reference, task):
    results = {}
    for metric in register_metric.values():
        metric(prediction, reference, results)
    task['metrics'] = results

def eval_data(data, data_type):
    results = data.copy()
    print(f'Starting {data_type} Evaluations!')
    for task in tqdm(results.values()):
        add_metrics(task['generated'], task['reference'], task)
    return results