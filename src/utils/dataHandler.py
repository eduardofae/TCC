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