from scipy.stats import pearsonr
from utils.dataHandler import read_data, write_data
from statistics import stdev

def get_human_evals(path, padding=1):
    with open(path, 'r', encoding='utf-8') as file:
        results = {
            'Consistência': [],
            'Naturalidade': [],
            'Relevância': [],
            'Coerência': []
        }
        lin_num = 0
        for line in file:
            if lin_num < padding: 
                lin_num += 1
                continue
            cols = line.split('\n')[0].split(',')[2:]
            for i, col in enumerate(cols):
                eval_type = list(results.keys())[i%4]
                if i//4 >= len(results[eval_type]):
                    results[eval_type].append({'scores': [], 'avg': 0})
                if col != '':
                    results[eval_type][i//4]['scores'].append(int(col))
                    results[eval_type][i//4]['avg'] = sum(results[eval_type][i//4]['scores']) / len(results[eval_type][i//4]['scores'])
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

def get_correlations(path):
    metrics = get_metrics(f'{path}/group.json')
    human_evals = get_human_evals(f'{path}/human_evals.csv')
    return {
        metric: { 
            eval_type: pearsonr(values, [val['avg'] for val in human_eval]) 
            for eval_type, human_eval in human_evals.items() 
        } 
        for metric, values in metrics.items()
    }

def get_metrics_avg(path):
    metrics = get_metrics(f'{path}/group.json')
    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)
    return metrics

def get_human_avgs(path):
    human_evals = get_human_evals(f'{path}/human_evals.csv')
    for k, val in human_evals.items():
        val_ = [v['avg'] for v in val]
        human_evals[k] = sum(val_) / len(val_)
    return human_evals

def get_human_stdevs(path):
    human_evals = get_human_evals(f'{path}/human_evals.csv')
    for k, val in human_evals.items():
        for i, v in enumerate(val):
            val[i] = stdev(v['scores'])
        human_evals[k] = sum(val) / len(val)
    return human_evals


RESULTS_PATHS = ['./Saídas Obtidas/QA/gemma-3-1b-it', './Saídas Obtidas/SUMM/ptt5-base-summ-xlsum']
for results_path in RESULTS_PATHS:
    write_data(get_correlations(results_path), f'{results_path}/correlations.json')
    write_data(get_metrics_avg(results_path) , f'{results_path}/metrics-avg.json' )
    write_data(get_human_avgs(results_path)  , f'{results_path}/human-avgs.json'  )
    write_data(get_human_stdevs(results_path), f'{results_path}/human-stdevs.json')