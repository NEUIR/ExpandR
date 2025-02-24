# LLM-QE: Improving Query Expansion by Aligning Large Language Models with Ranking Preferences

Source code for our paper :
LLM-QE: Improving Query Expansion by Aligning Large Language
Models with Ranking Preferences (Coming soon...)

## Overview
We introduce LLM-QE, a novel approach that leverages Large Language Models (LLMs) to generate document-based query expansions, thereby enhancing dense retrieval models. 

LLM-QE designs both rank-based and answer-based rewards and uses these reward models to optimize LLMs to align with the ranking preferences of both retrievers and LLMs, thus mitigating the hallucination of LLMs during query expansion. 
![method](assets/model.png)


## 1. Setup
(1) Use `git clone` to download this project:
```
git clone git@github.com:NEUIR/LLM-QE.git
cd LLM-QE
```
(2) Install the following packages using Pip or Conda under your environment
```
Python=3.10.14
torch=2.5.1
transformers==4.41.2
tqdm
trl==0.12.2
vllm==0.5.0.post1
accelerate==1.3.0
deepspeed==0.14.4
peft==0.11.1
jsonlines
```
(3) Install the modified `beir`:
```
cd src/beir
pip install -e .
```

## 2. Training LLM-QE:
You can download the lora checkpoints of LLM-QE directly from here and merge them, or follow the flow below to train LLM-QE.

### 2.1 Prepare the data
we use the public portion of dataset curated by authors of [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449). The dataset can be downloaded from the [GitHub page of Echo embeddings repository](https://github.com/jakespringer/echo-embeddings#training). To use the training script, the downloaded dataset should be placed in the `data` directory. The directory layout should be as follows:

```
data
├─ echo-data
    ├─ eli5_question_answer.jsonl
    ├─ fever.jsonl 
    ├─ hotpot_qa.jsonl
    ...
```
To merge these data, use the following command:
```
cd data/echo-data
cat *.jsonl > merge_data_80w.jsonl
```
Then run the following command to randomly split the data into two parts:
```
python LLM-QE/src/split.py
```

### 2.2 DPO Training
(1) First step: Download the related model

You need to download [lama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model as the vanilla Generation Model.

(2) Second step: Construct dpo training data

Then you can construct a dataset for dpo training by running this script, which includes multiple steps such as generating query expansion using LLM, reward model filtering data, and dividing the dataset.
```
cd LLM-QE/scripts
bash gen_dpo_data.sh
```
(3) Third step: Training the Generation Model

After constructing the training data, you can start training the query expansion generation model. 
```
bash dpo_train.sh
```
(4) Fourth step: Combine the weights

You need to combine the weights of the Generation model trained using lora in Third step.
```
bash merge_lora.sh
```
###  2.3 Supervised contrastive training
(1) First step: Download the related model

You need to download [Contriever](https://huggingface.co/facebook/contriever/tree/main) model as the vanilla retriever Model.

(2) Second step: Construct supervised contrastive training data

Then you can construct a dataset for supervised training by running this script, which includes generating query expansion using LLM and dividing the dataset.
```
bash gen_supervised_data.sh
```
(3) Third step: Training the retriever Model

After constructing the training data, you can start training the retriever model. 
```
bash supervised_train.sh
```

## 3. Evaluating LLM-QE
After training the LLM-QE model, you can test the performance of LLM-QE on Beir using the following command.

```
bash eval_beir_15.sh
```

## Contact
If you have questions, suggestions, and bug reports, please email:
```
ysj1426746590@outlook.com
```
