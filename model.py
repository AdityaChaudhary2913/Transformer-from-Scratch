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
    
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        
        # nn.Parameter is a special kind of Tensor, that will get 
        # automatically registered as Module's parameter
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff) # w1 & b1
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # w2 & b2
        
    def forward(self, x):
        agg_at_first_layer = self.linear1(x)
        activation_at_first_layer = torch.relu(agg_at_first_layer)
        activation_at_first_layer = self.dropout(activation_at_first_layer)
        agg_at_output_layer = self.linear2(activation_at_first_layer)
        return agg_at_output_layer
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # Shape Transformation: (batch_size, n_heads, seq_len, d_k) -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        scores = torch.nn.functional.softmax(scores, dim=-1) # (batch_size, n_heads, seq_len, d_k)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, value), scores
    
    def forward(self, query, key, value, mask=None):
        # Shape: (batch_size, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value) 
        
        # Shape Transformation: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.d_k).permute(0, 2, 1, 3)
    
        x, attention_scores = self.attention(q, k, v, mask, self.dropout)
        
        # Shape Transformation: (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float): 
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModeuleList([ResidualConnection(dropout) for _ in range(2)])
        
    # Shape of src_mask: (batch_size, seq_len, seq_len), apply to input of encoder
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.multi_head_attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, n_layers: int):
        super(Encoder, self).__init__()
        self.layers = n_layers
        self.norm = LayerNormalization()
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x