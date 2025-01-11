import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        
        self.sos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        src = item['translation'][self.lang_src]
        tgt = item['translation'][self.lang_tgt]
        src_enc = self.tokenizer_src.encode(src).ids
        tgt_enc = self.tokenizer_tgt.encode(tgt).ids
        return torch.tensor(src_enc), torch.tensor(tgt_enc)