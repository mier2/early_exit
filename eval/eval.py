import modeling_llama_wrappers_eval
import random
import math
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate
from torch.utils.data import DataLoader


from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import wandb


wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project="lora_early_exit_eval",
    )


if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )



model_checkpoint = "checkpoint-2500"

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



# MMLU Five-shot (Eval/Test only)
mmlu_dataset = load_dataset("json", data_files={
    'eval': 'data/mmlu/five_shot_mmlu_val.json',
    'test': 'data/mmlu/five_shot_mmlu_test.json',
})
# mmlu_dataset = mmlu_dataset.remove_columns('subject')
mmlu_dataset = mmlu_dataset['eval']

abcd_idx = [
tokenizer("A", add_special_tokens=False).input_ids[0],
tokenizer("B", add_special_tokens=False).input_ids[0],
tokenizer("C", add_special_tokens=False).input_ids[0],
tokenizer("D", add_special_tokens=False).input_ids[0],
]
data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=1024,
        target_max_len=256,
        train_on_source=False,
        predict_with_generate=False,
    )
data_loader = DataLoader(mmlu_dataset, collate_fn=data_collator, batch_size=1)
preds, refs = [], []
loss_mmlu = 0
accuracy = evaluate.load("accuracy")
comb_preds1, comb_preds2, comb_preds3 = [], [], []
preds_layer1, preds_layer2, preds_layer3, preds_layer4 = [], [], [], []
exit_layers = [8, 16, 24, 32]


for batch in tqdm(data_loader, total=len(data_loader)):
    labels = batch['labels']
    batch['return_dict'] = True
    batch['output_hidden_states'] = True
    batch['model_status'] = 'train'
    batch['exit_layers'] = [7, 15, 23, 31]
    
    outputs = model(**batch)
    hidden_states = outputs['hidden_states']
    orig_logits = outputs['logits']
    loss = outputs['loss']
    exit_layers_logits = list()
    for idx in exit_layers:
        exit_layers_logits.append(torch.nn.functional.softmax(model.lm_head(hidden_states[idx]), dim=-1))
    
    # input size: [exit_layer_num, batch_size, seq_len, vocab_dim] logits
    logits = torch.stack(exit_layers_logits, dim=0)
    # match merge_method:
        # case 0:
    lambda_const = 1
    N = logits.shape[2]
    temperature = 4
    threshold = [0.9 * lambda_const + 0.1 * math.exp(-temperature * t / N) for t in range(N)]
    threshold = torch.tensor(threshold, device=logits.device)[None, ...]
    exit_layer = (logits.shape[0]-1)*torch.ones(logits.shape[1:3],device=logits.device)
    initial_exit_layer = exit_layer
    for layer_num, layer_logit in enumerate(logits):
        test_val = torch.topk(layer_logit, k=2, dim=2)[0]
        test_val = (test_val[:,:,0] - test_val[:,:,1]).squeeze() 
        # if the top is greater than threshold and exit layer is still the last layer, modify to be current layer

        mask = (test_val > threshold) & (exit_layer == initial_exit_layer)
        exit_layer[mask] = layer_num
        #exit_layer[(test_val > threshold) and (exit_layer == initial_exit_layer)] = layer_num

    
    
    final_logits = torch.zeros((orig_logits[0].shape)).unsqueeze(dim=0)    # [batch_size, seq_len, vocab_dim]
    # print(f'final logits shape {final_logits.shape}')
    # print(f'logits shape {logits.shape}')
    for i in range(final_logits.shape[1]):
        exit_layer = exit_layer.long()
        final_logits[:, i, :] = torch.squeeze(logits[exit_layer[:,i]])[i, :]

        # case 1:
    final_logits_1 = torch.mean(logits, dim=0)

        # case 2:
    weights = [2/((logits.shape[0]+1)*4)*(n+1) for n in range(logits.shape[0])] # normalized weights
    weights = torch.tensor(weights, device=logits.device)
    final_logits_2 = torch.sum(weights[:, None, None, None] * logits, dim=0)

        # case 3:
    topk = torch.topk(logits, k=1, dim=0)[0].squeeze(dim=0)
    final_logits_3 = topk/torch.sum(topk, dim=2)[:,:,None]

    # output size: [batch_size, seq_len, vocab_dim] final_logits
    # logits = logits[0]
    exit_layers_preds = list() 
    
    for logit in exit_layers_logits:
        label_non_zero_id = (batch['labels'][0] != -100).nonzero()[0][0]
        logit_abcd = logit[0][label_non_zero_id-1][abcd_idx]
        exit_layers_preds.append(torch.argmax(logit_abcd).item())

    preds_layer1.append(exit_layers_preds[0])
    preds_layer2.append(exit_layers_preds[1])
    preds_layer3.append(exit_layers_preds[2])
    preds_layer4.append(exit_layers_preds[3])


    for i, logit in enumerate(final_logits):
        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
        preds.append(torch.argmax(logit_abcd).item())
    for i, logit in enumerate(final_logits_1):
        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
        comb_preds1.append(torch.argmax(logit_abcd).item())
    for i, logit in enumerate(final_logits_2):
        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
        comb_preds2.append(torch.argmax(logit_abcd).item())
    for i, logit in enumerate(final_logits_3):
        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
        comb_preds3.append(torch.argmax(logit_abcd).item())
    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
    refs += [abcd_idx.index(label) for label in labels.tolist()]
    loss_mmlu += loss.item()

results = {'mmlu_loss':loss_mmlu/len(data_loader)}
subject = mmlu_dataset['subject']
subjects = {s:{'refs':[], 'preds':[], 'preds_layer1': [], 'preds_layer2': [], 'preds_layer3': [], 'preds_layer4': [], 'preds_comb1': [], "preds_comb2": [], "preds_comb3": []} for s in set(subject)}
for s,p,r, pr1, pr2, pr3, pr4, comb1, comb2, comb3 in zip(subject, preds, refs, preds_layer1, preds_layer2, preds_layer3, preds_layer4, comb_preds1, comb_preds2, comb_preds3):
    subjects[s]['preds'].append(p)
    subjects[s]['refs'].append(r)
    subjects[s]['preds_layer1'].append(pr1)
    subjects[s]['preds_layer2'].append(pr2)
    subjects[s]['preds_layer3'].append(pr3)
    subjects[s]['preds_layer4'].append(pr4)
    subjects[s]['preds_comb1'].append(comb1)
    subjects[s]['preds_comb2'].append(comb2)
    subjects[s]['preds_comb3'].append(comb3)
subject_scores = []
subject_scores_layer1 = []
subject_scores_layer2 = []
subject_scores_layer3 = []
subject_scores_layer4 = []
subject_scores_comb1 = []
subject_scores_comb2 = []
subject_scores_comb3 = []
for subject in subjects:
    subject_score = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds']
    )['accuracy']
    subject_score_layer1 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_layer1']
    )['accuracy']
    subject_score_layer2 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_layer2']
    )['accuracy']
    subject_score_layer3 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_layer3']
    )['accuracy']
    subject_score_layer4 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_layer4']
    )['accuracy']
    subject_score_comb1 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_comb1']
    )['accuracy']
    subject_score_comb2 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_comb2']
    )['accuracy']
    subject_score_comb3 = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds_comb3']
    )['accuracy']
    results[f'mmlu_eval_accuracy_{subject}'] = subject_score
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_exitlayer1'] = subject_score_layer1
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_exitlayer2'] = subject_score_layer2
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_exitlayer3'] = subject_score_layer3
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_exitlayer4'] = subject_score_layer4
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_comb1'] = subject_score_comb1
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_comb2'] = subject_score_comb2
    # results[f'mmlu_{args.mmlu_split}_accuracy_{subject}_comb3'] = subject_score_comb3
    subject_scores.append(subject_score)
    subject_scores_layer1.append(subject_score_layer1)
    subject_scores_layer2.append(subject_score_layer2)
    subject_scores_layer3.append(subject_score_layer3)
    subject_scores_layer4.append(subject_score_layer4)
    subject_scores_comb1.append(subject_score_comb1)
    subject_scores_comb2.append(subject_score_comb2)
    subject_scores_comb3.append(subject_score_comb3)
results[f'mmlu_eval_accuracy'] = np.mean(subject_scores)
results[f'mmlu_eval_accuracy_exitlayer1'] = np.mean(subject_scores_layer1)
results[f'mmlu_eval_accuracy_exitlayer2'] = np.mean(subject_scores_layer2)
results[f'mmlu_eval_accuracy_exitlayer3'] = np.mean(subject_scores_layer3)
results[f'mmlu_eval_accuracy_exitlayer4'] = np.mean(subject_scores_layer4)
results[f'mmlu_eval_accuracy_comb1'] = np.mean(subject_scores_comb1)
results[f'mmlu_eval_accuracy_comb2'] = np.mean(subject_scores_comb2)
results[f'mmlu_eval_accuracy_comb3'] = np.mean(subject_scores_comb3)
wandb.log(results)

wandb.finish()