import torch
import torch.nn as nn
import math

# Matrix Representation: (Row Number, Column Number)

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model) # Shape: (batch_size, seq_len)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(seq_len, d_model) # Shape: (seq_len, d_model)
        
        # Shape of position:
        # Shape: (1, seq_len) before unsqueeze operation
        # Shape: (seq_len, 1) after unsqueeze operation as we want to 
        # expand it to (seq_len, d_model) by multiplying with div_term
        position = torch.arange(0, seq_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, seq_len, d_model) as we want to attain batch processing
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)