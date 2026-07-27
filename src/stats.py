from utils.dataHandler import write_data, read_data, get_metrics, get_human_evals
from scipy.stats import pearsonr, spearmanr 
from statistics import stdev
import krippendorff

def get_correlations(path):
    metrics = get_metrics(f'{path}/group.json')
    human_evals = get_human_evals(f'{path}/human_evals.csv')['Per_Instance']
    return {
        metric: { 
            eval_type: {
                'pearson': pearsonr(values, [val['avg'] for val in human_eval]),
                'spearman': spearmanr(values, [val['avg'] for val in human_eval])
            }
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
    human_evals = get_human_evals(f'{path}/human_evals.csv')['Per_Instance']
    for k, val in human_evals.items():
        val_ = [v['avg'] for v in val]
        human_evals[k] = sum(val_) / len(val_)
    return human_evals

def get_human_stdevs(path):
    human_evals = get_human_evals(f'{path}/human_evals.csv')['Per_Instance']
    for k, val in human_evals.items():
        for i, v in enumerate(val):
            val[i] = stdev(v['scores'])
        human_evals[k] = sum(val) / len(val)
    return human_evals

def get_krippendorff_alpha(path):
    human_evals = get_human_evals(f'{path}/human_evals.csv')['Per_Evaluator']
    results = {
        eval_type: krippendorff.alpha(reliability_data=evaluators, level_of_measurement='ordinal')
        for eval_type, evaluators in human_evals.items()
    }
    results['general'] = krippendorff.alpha(reliability_data=[[
            score 
            for eval_type in human_evals.keys() 
            for score in human_evals[eval_type][i]
        ]
        for i in range(len(human_evals['Consistência']))
    ], level_of_measurement='ordinal')
    return results

def get_output_lengths(path):
    data = read_data(f'{path}/group.json')
    lengths = []
    for d in data.values():
        lengths.append(len(d['generated'].split()))
    return {
        'avg': sum(lengths) / len(lengths),
        'stdev': stdev(lengths),
        'min': min(lengths),
        'max': max(lengths)
    }

def get_stats(task_results_path):
    write_data(get_correlations(task_results_path), f'{task_results_path}/correlations.json')
    write_data(get_metrics_avg(task_results_path), f'{task_results_path}/metrics-avg.json')
    write_data(get_human_avgs(task_results_path), f'{task_results_path}/human-avgs.json')
    write_data(get_human_stdevs(task_results_path), f'{task_results_path}/human-stdevs.json')
    write_data(get_krippendorff_alpha(task_results_path), f'{task_results_path}/krippendorff-alpha.json')
    write_data(get_output_lengths(task_results_path), f'{task_results_path}/output-lengths.json')

if __name__ == '__main__':
    PATHS = ['Saídas Obtidas/QA', 'Saídas Obtidas/SUMM']
    for path in PATHS:
        get_stats(path)