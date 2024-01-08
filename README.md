DoLa in text summarization!
===
## Introduction:

Recent advancements in Large Language Models (LLMs) have revolutionized the field of artificial intelligence, especially in natural language processing. However, a significant challenge persists in the form of inaccuracies and hallucinations in text generation. This project is inspired by the innovative approach of Decoding by Contrasting Layers (DoLa), as explored in the paper “DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models”. While DoLa demonstrates a way to reduce such errors, my project aims to migrate upon this foundation. In my project, I migrated this innovative approach to text summarization, based on the Xsum dataset and an encoder-decoder model “T5”.

## Method:

Recent language models consist of an embedding layer, N stacked transformer layers, and an affine layer φ(·) for predicting the next word distribution. The generated result will be processed by each transformer layer successively where the final output is from the mature layer. But, in DoLa, they proposed a variable called the early exit layer, which corresponded to the pre-mature layer. They obtain the outputs from the early exit layer (pre-mature layer) is called the hidden state. And implement φ(·) for each hidden state for predicting the pre-mature layer’s word distribution. To calculate the final answer, they compute the Jensen-Shannon Divergence (JSD) between the early exiting output distributions and the final layer output distribution and use a greedy algorithm to find a layer that has the most divergence between the mature layer and several pre-mature layers. After finding a layer with the most contrast, they used a logarithmic function to amplify the final output from DoLa.

## EXPERIMENTS:

In this project, we extend the Decoding by Contrasting Layers (DoLa) approach, previously implemented in decoder-based models like LLaMA and GPT, to enhance factual content in language tasks. We adapt DoLa for integration with the T5 model, an encoder-decoder framework, focusing on the summarization task in Xsum. Our methodology involves a comparative analysis using the Rouge metric to evaluate performance between the T5 baseline and its adaptation with DoLa. This study aims to not only measure the effectiveness of DoLa in improving summarization accuracy but also to explore the underlying mechanisms contributing to these improvements.

## Implementation:

To adapt the code with DoLa, they directly made some changes in the “transformers” package and used a customized “transformers” package. They changed the ‘forward’ function code in the “transformers” package to get the pre-mature layer’s word distribution. Add φ(·) for each hidden state to let them output word distribution. Then, they change the “greedy decoder” function code in the “utils” file which is also included in the “transformers” package. This greedy decoder achieves the goals of layer selection. 

In my implementation phase, I incorporated the DoLa method into the T5 model, focusing on layer utilization analysis. I developed an entry script to seamlessly load the T5 and LLaMA models along with the Xsum dataset. This setup involved configuring various parameters, utilizing the 'generate' function for output generation, and subsequently evaluating these outputs using the Rouge metric. This approach provided a structured framework to assess the model's performance and the efficacy of the DoLa integration in the context of language summarization tasks.

## Results:

I evaluate my model within the first 100 test data from Xsum dataset and use the model across LLaMA-7B, Flan-T5-base, Flan-T5-xxl compared by baseline and DoLa. The results show that LLaMA with DoLa has a better performance and Flan-T5 with DoLa cannot has a better performance compared to the baseline. 

![image](https://github.com/Xingyan3/DoLa_Xsum/assets/114642583/52fccdde-ced1-4981-83df-648eeb093d24)


## Analysis:

To find the reason why Flan-T5 with DoLa cannot have a better performance, I did some analysis to find out if Flan-T5 with DoLa performs some similar features shown in the DoLa paper. The first observation from the paper is factual knowledge tends to have a higher divergence in higher layers. This pattern indicates that the model is still changing its predictions in the last few layers, and potentially injecting more factual knowledge into the predictions. 

<img width="406" alt="image" src="https://github.com/Xingyan3/DoLa_Xsum/assets/114642583/7462b907-c22e-42ce-8c65-c6330ac4b9ca">


![image](https://github.com/Xingyan3/DoLa_Xsum/assets/114642583/05d8ad65-5051-4e00-a840-5a98fdcb2a82)

In my experiment, I didn’t find the similar trends shown above.

## Conclusion:

This project explored the integration of the Decoding by Contrasting Layers (DoLa) method into the T5 model, specifically targeting the Xsum summarization task. My implementation involved modifying the 'transformers' package to adapt the T5 and LLaMA models in the decoder for DoLa's contrasting layer approach. The comparative performance, assessed using the Rouge metric, revealed mixed results. While the LLaMA model showed improved performance with DoLa, the Flan-T5 adaptations did not yield better performance compared to their baselines. This outcome suggests that the effectiveness of DoLa may vary across different models and tasks. In addition, T5 is an encoder-decoder-based model. Thus, if we make some changes in the encoder, the results and conclusions may reverse.

# References:

[1] DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models

[2] ROUGE: A Package for Automatic Evaluation of Summaries

[3] Scaling Instruction-Finetuned Language Models

[4] https://huggingface.co/docs/transformers/index

[5] LLaMA: Open and Efficient Foundation Language Models

DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
===

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2309.03883-B21A1B)](https://arxiv.org/abs/2309.03883)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/YungSungChuang/status/1701623359153316255)
[![GitHub Stars](https://img.shields.io/github/stars/voidism/DoLa?style=social)](https://github.com/voidism/DoLa/stargazers)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voidism/DoLa/blob/master/dola_evaluation.ipynb)

Code for the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models"

Paper: https://arxiv.org/abs/2309.03883  
Authors: [Yung-Sung Chuang](https://people.csail.mit.edu/yungsung/) $^\dagger$, [Yujia Xie](https://sites.google.com/view/yujia) $^\ddagger$, [Hongyin Luo](https://luohongyin.github.io/) $^\dagger$, [Yoon Kim](https://people.csail.mit.edu/yoonkim/) $^\dagger$, [James Glass](https://people.csail.mit.edu/jrg/) $^\dagger$, [Pengcheng He](https://scholar.google.com/citations?user=TS1RoxAAAAAJ&hl=en) $^\ddagger$  
$^\dagger$ Massachusetts Institute of Technology, $^\ddagger$ Microsoft

## Overview

![DoLa](figure.png)

Despite their impressive capabilities, large language models (LLMs) are prone to hallucinations, i.e., generating content that deviates from  facts seen during pretraining. We propose a simple decoding strategy for reducing hallucinations with pretrained LLMs that does not require conditioning on retrieved external knowledge nor additional fine-tuning. Our approach obtains the next-token distribution by contrasting the differences in logits obtained from projecting the later layers versus earlier layers to the vocabulary space, exploiting the fact that factual knowledge in an LLMs has generally been shown to be localized to particular transformer layers. We find that this **D**ecoding by c**o**ntrasting **La**yers (DoLA) approach is able to better surface factual knowledge and reduce the generation of incorrect facts.  DoLA consistently improves the truthfulness across multiple choices tasks and open-ended generation tasks, for example improving performance of LLaMA family models on TruthfulQA by 12-17\% absolute points, demonstrating its potential in making LLMs reliably generate truthful facts.

## Setup

```
pip install -e transformers-4.28.1
pip install datasets
pip install accelerate
pip install openai # -> only for truthfulqa and gpt4_eval
```

## Experiments

### Arguments

| Argument          | Example           | Description   |
| ----------------- | ----------------- | ------------- |
| `--model-name`    | `huggyllama/llama-7b` | Specifies the model you want to use, currently we only support LLaMA-v1. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder. |
| `--output-path`   | `output-path.json` | Where to store the output results. |
| `--num-gpus`      | `1` | Number of GPUs to use, `1/2/4/8` for `7B/13B/30B/65B` model sizes respectively.  |
| `--max_gpu_memory`| `27` | Maximum GPU memory size (in GiB) to allocate. Default: 27 (for 32G V100).  |

### Understanding `--early-exit-layers`

The `--early-exit-layers` argument takes a string containing a sequence of layer numbers separated by commas, with no spaces in between. By specifying different number of layers, we make the model decode at different modes.


| Number of Layers Specified  | Example (str)     | Description of Decoding Mode                                                                                     |
| ---------------------------| ------------- | ----------------------------------------------------------------------------------------------- |
| 1                          | `-1`      | **Naive decoding** from the final layer output.       |
| 2                          | `16,32`   | **DoLa-static decoding** with the second specified layer (i.e. `32`) as the `mature_layer` and first specified layer (i.e. `16`) as `premature_layer`. |
| >2                         | `0,2,4,6,8,10,12,14,32`    | **DoLa decoding** with the last specified layer (i.e. `32`) as the `mature_layer` and all the preceding layers (i.e. `0,2,4,6,8,10,12,14`) as `candidate_premature_layers`. |

### FACTOR (Multiple Choices)
Please download the data file `wiki_factor.csv` from https://github.com/AI21Labs/factor

#### Baseline
```bash
python factor_eval.py --model-name huggyllama/llama-7b --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 1
python factor_eval.py --model-name huggyllama/llama-13b --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 2
python factor_eval.py --model-name huggyllama/llama-30b --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 4
python factor_eval.py --model-name huggyllama/llama-65b --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 8
```

#### DoLa
```bash
python factor_eval.py --model-name huggyllama/llama-7b --early-exit-layers 0,2,4,6,8,10,12,14,32 --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 1
python factor_eval.py --model-name huggyllama/llama-13b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,40 --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 2
python factor_eval.py --model-name huggyllama/llama-30b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,60 --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 4
python factor_eval.py --model-name huggyllama/llama-65b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,80 --data-path /path/to/wiki_factor.csv --output-path output-path.json --num-gpus 8
```

### TruthfulQA (Multiple Choices)

The `--data-path` should be a folder contains `TruthfulQA.csv`. If file not exists, it will be downloaded automatcially.

#### Baseline
```bash
python tfqa_mc_eval.py --model-name huggyllama/llama-7b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1
python tfqa_mc_eval.py --model-name huggyllama/llama-13b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2
python tfqa_mc_eval.py --model-name huggyllama/llama-30b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4
python tfqa_mc_eval.py --model-name huggyllama/llama-65b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8
```

#### DoLa
```bash
python tfqa_mc_eval.py --model-name huggyllama/llama-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1
python tfqa_mc_eval.py --model-name huggyllama/llama-13b --early-exit-layers 20,22,24,26,28,30,32,34,36,38,40 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2
python tfqa_mc_eval.py --model-name huggyllama/llama-30b --early-exit-layers 40,42,44,46,48,50,52,54,56,58,60 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4
python tfqa_mc_eval.py --model-name huggyllama/llama-65b --early-exit-layers 60,62,64,66,68,70,72,74,76,78,80 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8
```

### TruthfulQA

To evaluate the open-ended generation result of TruthfulQA, we need to finetune two GPT-3 curie models through OpenAI API:

```
openai api fine_tunes.create -t finetune_truth.jsonl -m curie --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
openai api fine_tunes.create -t finetune_info.jsonl -m curie --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
```

After finetuning, we can obtain the finetuned model names by `openai api fine_tunes.list | grep fine_tuned_model`.

Create a config file `gpt3.config.json` like this:

```json
{"gpt_info": "curie:ft-xxxxxxxxxx",
"gpt_truth": "curie:ft-xxxxxxxxxx",
"api_key": "xxxxxxx"}
```

Add the argument `--do-rating --gpt3-config gpt3.config.json` for GPT-3 evaluation.

#### Baseline
```bash
python tfqa_eval.py --model-name huggyllama/llama-7b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1 --do-rating --gpt3-config /path/to/gpt3.config.json
python tfqa_eval.py --model-name huggyllama/llama-13b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2 --do-rating --gpt3-config /path/to/gpt3.config.json
python tfqa_eval.py --model-name huggyllama/llama-30b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4 --do-rating --gpt3-config /path/to/gpt3.config.json
python tfqa_eval.py --model-name huggyllama/llama-65b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8 --do-rating --gpt3-config /path/to/gpt3.config.json
```

#### DoLa
```bash
python tfqa_eval.py --model-name huggyllama/llama-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1 --do-rating --gpt3-config /path/to/gpt3.config.json
python tfqa_eval.py --model-name huggyllama/llama-13b --early-exit-layers 20,22,24,26,28,30,32,34,36,38,40 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2 --do-rating --gpt3-config /path/to/gpt3.config.json
python tfqa_eval.py --model-name huggyllama/llama-30b --early-exit-layers 40,42,44,46,48,50,52,54,56,58,60 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4 --do-rating --gpt3-config /path/to/gpt3.config.json
python tfqa_eval.py --model-name huggyllama/llama-65b --early-exit-layers 60,62,64,66,68,70,72,74,76,78,80 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8 --do-rating --gpt3-config /path/to/gpt3.config.json
```

### GSM8K

We use a random sampled subset of GSM8K training set to serve as the validation set of both StrategyQA and GSM8K. The file can be downloaded [here](https://www.dropbox.com/scl/fi/o8zde51x0erejbp8bo8d5/gsm8k-train-sub.jsonl?rlkey=yu90sjt58fk1cell3ey4v8xuu&dl=0).

The `--data-path` argument should be either:
- A folder that contains `gsm8k_test.jsonl`, otherwise the files will be downloaded automatically to the folder you specified. 
- (Only for GSM8K) The path to `gsm8k-train-sub.jsonl` which can be downloaded by the link above.

#### Baseline
```bash
python gsm8k_eval.py --model-name huggyllama/llama-7b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1
python gsm8k_eval.py --model-name huggyllama/llama-13b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2
python gsm8k_eval.py --model-name huggyllama/llama-30b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4
python gsm8k_eval.py --model-name huggyllama/llama-65b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8
```

#### DoLa
```bash
python gsm8k_eval.py --model-name huggyllama/llama-7b --early-exit-layers 0,2,4,6,8,10,12,14,32 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1
python gsm8k_eval.py --model-name huggyllama/llama-13b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,40 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2
python gsm8k_eval.py --model-name huggyllama/llama-30b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,60 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4
python gsm8k_eval.py --model-name huggyllama/llama-65b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,80 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8
```

### StrategyQA

The `--data-path` argument should be a folder that contains `strategyqa_train.json`, otherwise the files will be downloaded automatically to the folder you specified.

#### Baseline
```bash
python strqa_eval.py --model-name huggyllama/llama-7b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1
python strqa_eval.py --model-name huggyllama/llama-13b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2
python strqa_eval.py --model-name huggyllama/llama-30b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4
python strqa_eval.py --model-name huggyllama/llama-65b --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8
```

#### DoLa
```bash
python strqa_eval.py --model-name huggyllama/llama-7b --early-exit-layers 0,2,4,6,8,10,12,14,32 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 1
python strqa_eval.py --model-name huggyllama/llama-13b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,40 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 2
python strqa_eval.py --model-name huggyllama/llama-30b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,60 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 4
python strqa_eval.py --model-name huggyllama/llama-65b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,80 --data-path /path/to/data/folder --output-path output-path.json --num-gpus 8
```

### GPT-4 Evaluation (Vicuna QA Benchmark)

In GPT-4 evaluation, we need the question file from [FastChat](https://github.com/lm-sys/FastChat). In the following commands, we assume the path to your FastChat repo is `$fastchat`.

#### Baseline
```bash
python gpt4_judge_eval.py --model-name huggyllama/llama-7b --model-id llama-7b-baseline --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 1
python gpt4_judge_eval.py --model-name huggyllama/llama-13b --model-id llama-13b-baseline --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 2
python gpt4_judge_eval.py --model-name huggyllama/llama-30b --model-id llama-30b-baseline --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 4
python gpt4_judge_eval.py --model-name huggyllama/llama-65b --model-id llama-65b-baseline --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 8
```

#### DoLa
```bash
python gpt4_judge_eval.py --model-name huggyllama/llama-7b --early-exit-layers 0,2,4,6,8,10,12,14,32 --model-id llama-7b-dola --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 1
python gpt4_judge_eval.py --model-name huggyllama/llama-13b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,40 --model-id llama-13b-dola --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 2
python gpt4_judge_eval.py --model-name huggyllama/llama-30b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,60 --model-id llama-30b-dola --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 4
python gpt4_judge_eval.py --model-name huggyllama/llama-65b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,80 --model-id llama-65b-dola --question-file $fastchat/eval/table/question.jsonl --answer-file output-answer.jsonl --num-gpus 8
```

After running the above commands to generate the model responses, we need OpenAI API key to pairwise compare the responses from different decoding results.

```bash
python $fastchat/eval/eval_gpt_review.py -q $fastchat/eval/table/question.jsonl -a output-answer-1.jsonl output-answer-2.jsonl -p $fastchat/eval/table/prompt.jsonl -r $fastchat/eval/table/reviewer.jsonl -o output-review-path.jsonl -k openai_api_key
```

For more details of GPT-4 evaluation, please check [vicuna-blog-eval](https://github.com/lm-sys/vicuna-blog-eval/tree/main/eval).

## Reference Repositories
- FastChat: https://github.com/lm-sys/FastChat
- ContrastiveDecoding: https://github.com/XiangLi1999/ContrastiveDecoding
- TruthfulQA: https://github.com/sylinrl/TruthfulQA
- zero_shot_cot: https://github.com/kojima-takeshi188/zero_shot_cot
- FederatedScope: https://github.com/alibaba/FederatedScope

## Citation

[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2309.03883-green?color=FF8000?color=009922)](https://doi.org/10.48550/arXiv.2309.03883)

Please cite our paper if it's helpful to your work!
```
@article{chuang2023dola,
  title={DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models},
  author={Chuang, Yung-Sung and Xie, Yujia and Luo, Hongyin and Kim, Yoon and Glass, James and He, Pengcheng},
  journal={arXiv preprint arXiv:2309.03883},
  year={2023},
}
```
