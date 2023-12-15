import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
from tqdm import tqdm
import torch
import os
from dola_copy import DoLa
from prompt import build_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda"
num_gpus = 1
repetition_penalty = 1.5
n_samples = 1
start_sample = 4
mode = "dola"
mode = "baseline"

# set dola
if mode == "dola":
    # early_exit_layers = "0,2,4,6,8,10,12,14,32"
    early_exit_layers = "0,2,4,6,8,12"
    early_exit_layers = [int(x) for x in early_exit_layers.split(',')]
    print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
    mature_layer = early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = early_exit_layers[:-1]
    premature_layer_dist = {l:0 for l in candidate_premature_layers}

# set baseline
if mode == "baseline":
    early_exit_layers = None
    mature_layer = None
    candidate_premature_layers = None
    premature_layer_dist = None
    premature_layer = None

# ini params
max_new_tokens = 256
do_sample = False
top_p = 0.95
top_k = 0
temperature = 0.9
relative_top = 0.1

# 1. load model and tokenizer
model_name = 'google/flan-t5-base'
# model_name = 'spacemanidol/flan-t5-large-xsum'
# model_name = 't5-base'
# model_name = 'huggyllama/llama-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DoLa(model_name, device, num_gpus)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# move to gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# 2. load dataset
# with open("test.txt", "r") as file:  # read from file
#     test_data = file.readlines()
data_set = load_dataset('xsum')
test_data = data_set['test']
# data_set = load_dataset('csv', data_files='wikihowAll.csv')
# test_data = data_set['train']

# 3. generate
summaries = []
print(data_set)
for article in tqdm(test_data["document"][start_sample:start_sample+n_samples], desc="Processing articles"):
    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, repetition_penalty=repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=relative_top)
    model_completion, c_dist = model.generate("summerize: " + article, **generate_kwargs)
    print(article)
    if mode == "dola":
        for k, v in c_dist.items():
            premature_layer_dist[k] += v
    
    # encoded_article = tokenizer.encode(article, return_tensors="pt", max_length=4096, truncation=True).to(device)
    # summary_ids = model.generate(encoded_article)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(model_completion)

if mode == "dola":
    total_tokens = sum(premature_layer_dist.values())
    if total_tokens > 0:
        for l in candidate_premature_layers:
            print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))

# 4. save results
with open("results.json", "w") as file:
    json.dump(summaries, file)

print("Summaries saved to results.json!")

rouge = load_metric("rouge")

# rouge evaluation
references = test_data["summary"][start_sample:start_sample+n_samples]
print(references)
results = rouge.compute(predictions=summaries, references=references)

for key, value in results.items():
    print(f"{key}: {value.mid.fmeasure:.4f}") 