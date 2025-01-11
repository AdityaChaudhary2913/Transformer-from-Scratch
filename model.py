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
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    # Shape of src_mask: (batch_size, seq_len, seq_len), apply to input of encoder
    # Shape of tgt_mask: (batch_size, seq_len, seq_len), apply to input of decoder
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, n_layers: int):
        super(Decoder, self).__init__()
        self.layers = n_layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Shape Transformation: 
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask) # (batch_size, seq_len, d_model)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask) # (batch_size, seq_len, d_model)

    def project(self, x):
        return self.projection_layer(x) # (batch_size, seq_len, vocab_size)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, dff=2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)
        
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer =  Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer