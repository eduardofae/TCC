from tasks.summarization import run_summ
from tasks.questionAnswering import run_qa
from stats import get_stats

task_paths = [
    run_summ(),
    run_qa()
]

for path in task_paths:
    get_stats(path)