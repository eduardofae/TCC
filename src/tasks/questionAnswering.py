from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from confidential import TOKEN
from random import sample

model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

model_pt = pipeline("text-generation", model=model, tokenizer=tokenizer)

INSTRUCTIONS="<start_of_turn>system\nDado o contexto abaixo, responda a pergunta solicitada. Retorne apenas a resposta.<end_of_turn>\n"

def gen_answer(question, context):
    return model_pt(f"{INSTRUCTIONS}<start_of_turn>user\nContexto: {context}\nPergunta: {question}\n<end_of_turn>\n<start_of_turn>model\n", max_new_tokens=100)[0]['generated_text'].split('<start_of_turn>model\n')[1]

from datasets import load_dataset
def get_tasks():
    dataset = load_dataset("benjleite/FairytaleQA-translated-ptBR", split='test', token=TOKEN)
    ids = sample(range(len(dataset['story_section'])), 40)
    contexts = dataset['story_section']
    questions = dataset['question']
    references = dataset['answer']
    return {
        questions[i]: {
            'context': context,
            'reference': references[i]
        }
        for i, context in enumerate(contexts) 
        if i in ids
    }

from metrics import eval_data
from utils.dataHandler import write_data, get_output_paths
from tqdm import tqdm
from groups import gen_all_groups
TASK_NAME = 'qa'
DATASET_NAME = 'FairytaleQA-translated-ptBR'
MODEL_NAME = model_name.split('/')[1]
OUTPUT_PATHS = get_output_paths(TASK_NAME, DATASET_NAME, MODEL_NAME)
def run_qa():
    tasks = get_tasks()
    data = tasks.copy()
    data_type = 'Answer'
    print(f'Starting {data_type} Generation!')
    for task in tqdm(tasks):
        data[task]['generated'] = gen_answer(task, data[task]['context'])
    write_data(data, OUTPUT_PATHS['results'])
    write_data(eval_data(data, data_type), OUTPUT_PATHS['metrics'])
    gen_all_groups(OUTPUT_PATHS['folder'])
