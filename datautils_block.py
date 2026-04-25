from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import torch
from torch.utils.data import Dataset
import os
from pathlib import Path


LOCAL_REDPAJAMA_PARQUET_DIR = os.environ.get(
    "EFFICIENTQAT_REDPJ_LOCAL_DIR",
    "/root/autodl-tmp/EfficientQAT/cache/redpajama_v2_sample_parquet",
)
LOCAL_WIKITEXT2_PARQUET_DIR = os.environ.get(
    "EFFICIENTQAT_WIKITEXT2_LOCAL_DIR",
    "/root/autodl-tmp/EfficientQAT/cache/wikitext2_official",
)


def _load_local_redpajama_split(split: str):
    parquet_dir = Path(LOCAL_REDPAJAMA_PARQUET_DIR)
    if not parquet_dir.exists():
        return None
    train_files = sorted(str(path) for path in parquet_dir.glob("train-*.parquet"))
    val_files = sorted(str(path) for path in parquet_dir.glob("validation-*.parquet"))
    data_files = {}
    if train_files:
        data_files["train"] = train_files
    if val_files:
        data_files["validation"] = val_files
    if split not in data_files:
        return None
    print(f"load local redpajama parquet split={split} from {parquet_dir}")
    return load_dataset("parquet", data_files=data_files, split=split)


def _load_local_wikitext2_split(split: str):
    parquet_dir = Path(LOCAL_WIKITEXT2_PARQUET_DIR)
    parquet_path = parquet_dir / f"{split}.parquet"
    if not parquet_path.exists():
        return None
    print(f"load local wikitext2 parquet split={split} from {parquet_path}")
    return load_dataset("parquet", data_files={split: str(parquet_path)}, split=split)

def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")

    traindata = _load_local_wikitext2_split("train")
    testdata = _load_local_wikitext2_split("test")
    if traindata is None:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    if testdata is None:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # 拼接
    train_text = "\n\n".join(traindata['text'])
    test_text = "\n\n".join(testdata['text'])

    trainenc = tokenizer(train_text, return_tensors='pt')
    testenc = tokenizer(test_text, return_tensors='pt')

    if test_only:
        return testenc

    random.seed(seed)

    trainloader = []
    for _ in tqdm(range(train_size)):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in tqdm(range(val_size)):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    try:
        # set local path for faster loading
        traindata = load_dataset("arrow",
                    data_files={
                        "train": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-6fbe877195f42de5/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-train-00000-of-00002.arrow",
                        "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                    },split='train'
                    )
        valdata = load_dataset("arrow",
                    data_files={
                        "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                    },split='validation'
                    )
    except:
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc 

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))



    return trainloader, valloader 

def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama")
    local_train = _load_local_redpajama_split("train")
    if local_train is not None:
        traindata = local_train
    else:
        try:
            loacal_dataset = "/cpfs01/user/chenmengzhao/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample"
            traindata = load_dataset(loacal_dataset,split='train')   
        except:
            traindata = load_dataset("togethercomputer/RedPajama-Data-1T",split='train')   
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed) 
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader



def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'c4' in name:
        return get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer,train_size,val_size,seed,seqlen)
    else:
        raise NotImplementedError



@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        if hasattr(model,'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}:{ppl}')
        results[dataset] = ppl.item()
    model.config.use_cache = use_cache
    return results

class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/block_training_data', off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size%batch_size == 0
         
        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros((self.size//self.batch_size, self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)

    def _initialize_data_on_disk(self):
        for idx in range(self.size//self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size//self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data
