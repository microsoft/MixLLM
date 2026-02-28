# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import numpy as np
import torch

# def set_seed(seed):
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)


class DataUtils:

    @staticmethod
    def get_wikitext2(nsamples, seed, seqlen, model):
        from datasets import load_dataset
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer("\n\n".join(traindata['text']),
                             return_tensors='pt')
        dataset_eval = tokenizer("\n\n".join(testdata['text']),
                                 return_tensors='pt')

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({
                "input_ids": inp,
                "attention_mask": attention_mask
            })
        return trainloader, dataset_eval

    @staticmethod
    def get_ptb(nsamples, seed, seqlen, model):
        from datasets import load_dataset
        traindata = load_dataset('ptb_text_only',
                                 'penn_treebank',
                                 split='train')
        valdata = load_dataset('ptb_text_only',
                               'penn_treebank',
                               split='validation')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer("\n\n".join(traindata['sentence']),
                             return_tensors='pt')
        dataset_eval = tokenizer("\n\n".join(valdata['sentence']),
                                 return_tensors='pt')

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({
                "input_ids": inp,
                "attention_mask": attention_mask
            })
        return trainloader, dataset_eval

    @staticmethod
    def get_c4(nsamples, seed, seqlen, model):
        from datasets import load_dataset
        traindata = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train',
            verification_mode='no_checks')
        valdata = load_dataset('allenai/c4',
                               data_files={
                                   'validation':
                                       'en/c4-validation.00000-of-00008.json.gz'
                               },
                               split='validation',
                               verification_mode='no_checks')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                # if trainenc.input_ids.shape[1] >= seqlen:
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({
                "input_ids": inp,
                "attention_mask": attention_mask
            })

        import random
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                # if tmp.input_ids.shape[1] >= seqlen:
                if tmp.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)

        class TokenizerWrapper:

            def __init__(self, input_ids):
                self.input_ids = input_ids

        valenc = TokenizerWrapper(valenc)

        return trainloader, valenc

    @staticmethod
    def get_ptb_new(nsamples, seed, seqlen, model):
        from datasets import load_dataset
        traindata = load_dataset('ptb_text_only',
                                 'penn_treebank',
                                 split='train')
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer(" ".join(traindata['sentence']),
                             return_tensors='pt')
        dataset_eval = tokenizer(" ".join(testdata['sentence']),
                                 return_tensors='pt')

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({
                "input_ids": inp,
                "attention_mask": attention_mask
            })
        return trainloader, dataset_eval

    @staticmethod
    def get_c4_new(nsamples, seed, seqlen, model):
        from datasets import load_dataset
        traindata = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train')
        valdata = load_dataset('allenai/c4',
                               data_files={
                                   'validation':
                                       'en/c4-validation.00000-of-00008.json.gz'
                               },
                               split='validation')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                # if trainenc.input_ids.shape[1] >= seqlen:
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({
                "input_ids": inp,
                "attention_mask": attention_mask
            })

        valenc = tokenizer(' '.join(valdata[:1100]['text']),
                           return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:

            def __init__(self, input_ids):
                self.input_ids = input_ids

        valenc = TokenizerWrapper(valenc)

        return trainloader, valenc

    @staticmethod
    def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
        if 'wikitext2' in name:
            return DataUtils.get_wikitext2(nsamples, seed, seqlen, model)
        if 'ptb' in name:
            if 'new' in name:
                return DataUtils.get_ptb_new(nsamples, seed, seqlen, model)
            return DataUtils.get_ptb(nsamples, seed, seqlen, model)
        if 'c4' in name:
            if 'new' in name:
                return DataUtils.get_c4_new(nsamples, seed, seqlen, model)
            return DataUtils.get_c4(nsamples, seed, seqlen, model)
        raise ValueError(f"Unknown dataset: {name}.")

    @staticmethod
    def trainloader_to_tensor(trainloader, device='cpu'):
        datalist = [inp['input_ids'].to(device) for inp in trainloader]
        datatensor = torch.cat(datalist, dim=0)
        return datatensor, datalist
