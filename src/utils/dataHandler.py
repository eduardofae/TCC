import json
def read_data(path):
    with open(path, 'r', encoding='utf8') as file:
        return json.load(file)

def write_data(results, path):
    with open(path, 'w', encoding='utf8') as file:
        file.write(json.dumps(results))

from pathlib import Path
def get_output_paths(task_name, dataset_name, model_name):
    path = f'outputs/{task_name}/{dataset_name}/{model_name}'
    Path(path).mkdir(parents=True, exist_ok=True)
    return {
        'results': f'{path}/results.json',
        'metrics': f'{path}/metrics.json',
        'folder': path
    }

import numpy as np
QUALITY_DIMENSIONS = ['Consistência', 'Naturalidade', 'Relevância', 'Coerência']
ORDERING_TYPES = ['Per_Evaluator', 'Per_Instance']

def get_human_evals(path, padding=1):
    results = {
        ordering: {
            dimension: [] 
            for dimension in QUALITY_DIMENSIONS
        } 
        for ordering in ORDERING_TYPES
    }
    with open(path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file):
            if line_num < padding: continue
            cols = line.strip().split(',')[2:]
            for col_num, col in enumerate(cols):
                eval_type = QUALITY_DIMENSIONS[col_num%4]

                res_per_inst = results['Per_Instance'][eval_type]
                if col_num // 4 >= len(res_per_inst):
                    res_per_inst.append({'scores': [], 'avg': 0})
                if col != '':
                    cur_inst = res_per_inst[col_num//4]
                    cur_inst['scores'].append(int(col))
                    cur_inst['avg'] = sum(cur_inst['scores']) / len(cur_inst['scores'])

                res_per_eval = results['Per_Evaluator'][eval_type]
                if col_num < 4:
                    res_per_eval.append([])
                res_per_eval[line_num-padding].append(int(col) if col != '' else np.nan)
    return results

def get_metrics(path):
    data = read_data(path)
    metrics = {}
    for d in data.values():
        mtrcs = d['metrics']
        for key in mtrcs:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(mtrcs[key])
    return metrics