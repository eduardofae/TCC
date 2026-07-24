from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration

token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
model_name = 'recogna-nlp/ptt5-base-summ-xlsum'

tokenizer = T5Tokenizer.from_pretrained(token_name)
model_pt = T5ForConditionalGeneration.from_pretrained(model_name)

def gen_summary(text):
    inputs = tokenizer.encode(text, max_length=512, truncation=True, return_tensors='pt')
    summary_ids = model_pt.generate(inputs, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

from datasets import load_dataset
from confidential import TOKEN
def get_tasks():
    dataset = load_dataset("arubenruben/cnn_dailymail_azure_pt_pt", split='test[:500]', token=TOKEN)
    texts = dataset['document']
    references = dataset['summary']
    titles = [data.split('.')[0] for data in dataset['document']]
    return {
        titles[i]: {
            'text': text,
            'reference': references[i]
        }
        for i, text in enumerate(texts)
    }

from metrics import eval_data
from utils.dataHandler import write_data, get_output_paths
from tqdm import tqdm
from groups import gen_all_groups
TASK_NAME = 'summarization'
DATASET_NAME = 'cnn_dailymail_azure_pt_pt'
MODEL_NAME = model_name.split('/')[1]
OUTPUT_PATHS = get_output_paths(TASK_NAME, DATASET_NAME, MODEL_NAME)
def run_summ():
    data = get_tasks()
    data_type = 'Summary'
    print(f'Starting {data_type} Generation!')
    for task in tqdm(data.values()):
        task['generated'] = gen_summary(task['text'])
    write_data(data, OUTPUT_PATHS['results'])
    write_data(eval_data(data, data_type), OUTPUT_PATHS['metrics'])
    gen_all_groups(OUTPUT_PATHS['folder'])