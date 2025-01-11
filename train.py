import torch
import torch.nn as nn
from torch .utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = f"{config['data_dir']}/{lang}_tokenizer.json"
    if not Path(tokenizer_path).exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)  
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def  get_ds(config):
    ds_raw = load_dataset("opus_books", f"{config['lamg_src']}-{config['lang_tgt']}", split = 'train')
    
    tokeninzer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokeninzer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds, val_ds = random_split(ds_raw, [train_ds_size, val_ds_size])