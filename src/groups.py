from pathlib import Path
from utils.dataHandler import read_data, write_data
from random import sample

def gen_all_groups(path):
    folder = Path(path)
    file_name = "metrics.json"
    files = list(folder.rglob(file_name))

    for file in files:
        data = read_data(file)
        data = { k:v for k,v in data.items() if len(v['context'] if 'context' in v else v['text']) <= 2000 }
        data = { k:v for k,v in data.items() if 'não consigo' not in v['reference'].lower() and 'não possuo' not in v['reference'].lower()}
        metrics = dict(sample(list(data.items()), 40))
        write_data(metrics, f'{file.parent}/group.json')
        print_groups(file.parent)

def print_groups(path):
    folder = Path(path)
    file_name = "group.json"
    files = list(folder.rglob(file_name))

    for file in files:
        data = read_data(file)
        str_groups = ''
        for k, v in data.items():
            str_groups += f'Title: {k}\n\nText: {v['context'] if 'context' in v else v['text']}\n\nAnswer: {v['generated']}\n\n================================================================\n\n'
        with open(f'{file.parent}/group.txt', 'w', encoding='utf8') as file:
            file.write(str_groups)