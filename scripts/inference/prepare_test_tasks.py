
import json
import sys
sys.path.append(".")

from src import config

tasks_path = "data/test_data/tasks.json"
splits_path = config.TEST_PHASE_TASKS_PATH

d_test_task = json.load(open(tasks_path, "r"))
print(d_test_task["monolingual"]["ara"].keys())

for lang in config.TEST_PHASE_LANGS:
    print(lang)
    print(d_test_task["monolingual"][lang].keys())
    d_test_task["monolingual"][lang]["posts_train"] = []
    d_test_task["monolingual"][lang]["posts_dev"] = d_test_task["monolingual"][lang]["posts_test"]

d_test_task["crosslingual"]["posts_train"] = []
d_test_task["crosslingual"]["posts_dev"] = d_test_task["crosslingual"]["posts_test"]

d_test_task["crosslingual"].keys()

json.dump(d_test_task, open(splits_path, "w"))