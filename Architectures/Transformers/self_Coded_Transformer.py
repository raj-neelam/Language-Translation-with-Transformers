import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_positional_encoding(context, dimension):
    pos = np.arange(context)[:, np.newaxis]
    div_term = 1 / (10000 ** (np.arange(0, dimension, 2) / dimension))
    
    enc = np.zeros((context, dimension))
    enc[:, 0::2] = np.sin(pos * div_term)
    enc[:, 1::2] = np.cos(pos * div_term)
    return enc

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0 # head should evenly devide embeding dimention

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x,  pad_mask=None):
        B, C, E = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, C, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if pad_mask is not None:
            # pad_mask: (B, C) → (B, 1, 1, C)
            scores = scores.masked_fill(
                ~pad_mask[:, None, None, :],
                -1e9
            )

        attn = scores.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, C, E)

        return self.out_proj(out)

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, pad_mask=None):
        B, C, E = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, C, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # MASKING HAPPENS HERE
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).bool()
        scores = scores.masked_fill(~causal_mask, -1e9)

        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask[:, None, None, :], -1e9)

        attn = scores.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, C, E)

        return self.out_proj(out)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, decoder_x, encoder_x, encoder_pad_mask=None):
        B, C_tgt, E = decoder_x.shape
        _, C_src, _ = encoder_x.shape

        q = self.q_proj(decoder_x)
        k = self.k_proj(encoder_x)
        v = self.v_proj(encoder_x)

        # reshape → (B, H, C, D)
        q = q.reshape(B, C_tgt, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, C_src, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, C_src, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if encoder_pad_mask is not None:  # [change]
            scores = scores.masked_fill(
                ~encoder_pad_mask[:, None, None, :], -1e9
            )

        attn = scores.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, C_tgt, E)

        return self.out_proj(out)

class Transformer_Encoder_Block(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.2): # inp_dim is vocab size of the english tokenizer
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim, 3*embed_dim),
            nn.ReLU(),
            nn.Linear(3*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, words_vec, pad_mask=None):  
        informed_words = self.mhsa(words_vec, pad_mask)   # (B, C, E) Attention
        informed_words = self.dropout1(informed_words) # droupout

        words_vec = words_vec + informed_words
        words_vec = self.norm1(words_vec) # (B, C, E) Add & Normalize

        thought_words = self.ffnn(words_vec) # feef forward neural network

        thought_words = self.dropout2(thought_words) # droupout
        words_vec = words_vec + thought_words
        words_vec = self.norm2(words_vec) # (B, C, E) Add & Normalize

        return words_vec

class Full_Encoder(nn.Module):
    def __init__(self, input_vocab, context_length, embed_dim, num_layers=6, num_heads=8, dropout_rate=0.2, use_learnable_positional_encoding=False, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.word_embedding = nn.Embedding(input_vocab, embed_dim, padding_idx=pad_id)
        self.use_learnable_positional_encoding = use_learnable_positional_encoding
        if not use_learnable_positional_encoding:
            pe = torch.tensor(
                create_positional_encoding(context_length, embed_dim),
                dtype=torch.float32
            )
            self.register_buffer("positional_encoding", pe) # register and save positional encodings
        else:
            self.positional_encoding = nn.Parameter(torch.randn(context_length, embed_dim))
        
        self.blocks = nn.ModuleList(
            [Transformer_Encoder_Block(embed_dim, num_heads, dropout_rate) for _ in range(num_layers)]
        )

    def forward(self, x, pad_mask=None):                             # (B, C)
        embedded = self.word_embedding(x)               # (B, C, E) batch, context_length, embedding
        words_vec = self.positional_encoding[:embedded.size(1)] + embedded   # (B, C, E) Adding positonal embeddings
        
        pad_mask = (x != self.pad_id) 

        for block in self.blocks:
            words_vec = block(words_vec, pad_mask)
        
        if pad_mask is not None:
            words_vec = words_vec * pad_mask.unsqueeze(-1)

        return words_vec

class Transformer_Decoder_Block(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.2):
        super().__init__()
        self.attention = MaskedMultiHeadSelfAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 3 * embed_dim),
            nn.ReLU(),
            nn.Linear(3 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, word_vec, encoder_output, decoder_pad_mask=None, encoder_pad_mask=None):
        informed_words = self.attention(word_vec, decoder_pad_mask)
        informed_words = self.dropout1(informed_words)
        word_vec = self.norm1(informed_words + word_vec)

        informed_words = self.cross_attention(word_vec, encoder_output, encoder_pad_mask)
        informed_words = self.dropout2(informed_words)
        word_vec = self.norm2(informed_words + word_vec)

        informed_words = self.feed_forward(word_vec)
        informed_words = self.dropout3(informed_words)
        word_vec = self.norm3(informed_words + word_vec)
        
        return word_vec

class Full_Decoder(nn.Module):
    def __init__(self, input_vocab, context_length, embed_dim, num_layers=6, num_heads=8, dropout_rate=0.2, use_learnable_positional_encoding=False, pad_id=0): # hindi input vocab size
        super().__init__()
        self.pad_id = pad_id
        self.input_embedding = nn.Embedding(input_vocab, embed_dim, padding_idx=pad_id)
        self.use_learnable_positional_encoding = use_learnable_positional_encoding
        if not use_learnable_positional_encoding:
            pe = torch.tensor(
                create_positional_encoding(context_length, embed_dim),
                dtype=torch.float32
            )
            self.register_buffer("positional_encoding", pe) # register and save positional encodings
        else:
            self.positional_encoding = nn.Parameter(torch.randn(context_length, embed_dim))
        
        self.blocks = nn.ModuleList(
            [Transformer_Decoder_Block(embed_dim, num_heads, dropout_rate) for _ in range(num_layers)]
        )
        self.unembedding = nn.Linear(embed_dim, input_vocab)

    def forward(self, x, encoder_output, encoder_tokens):                             # (B, C)
        inp_embedded = self.input_embedding(x)               # (B, C, E) batch, context_length, embedding
        inp_words_vec = self.positional_encoding[:inp_embedded.size(1)] + inp_embedded   # (B, C, E) Adding positonal embeddings
        
        decoder_pad_mask = (x != self.pad_id)                # [change]
        encoder_pad_mask = (encoder_tokens != self.pad_id)   # [change]

        for block in self.blocks:
            inp_words_vec = block(inp_words_vec, encoder_output, decoder_pad_mask, encoder_pad_mask)
        
        inp_words_vec = inp_words_vec * decoder_pad_mask.unsqueeze(-1)
        
        inp_words_vec_logits = self.unembedding(inp_words_vec)
        return inp_words_vec_logits

class Self_Coded_Language_Transformer(nn.Module):
    def __init__(self, 
    encoder_language_vocab, 
    decoder_language_vocab, 
    context_length,
    embed_dim, 
    num_layers=6, 
    num_heads=8, 
    dropout_rate=0.2,
    pad_id=0,
    use_learnable_positional_encoding=False
    ):
        super().__init__()
        self.encoder = Full_Encoder(encoder_language_vocab, context_length, embed_dim, num_layers, num_heads, dropout_rate, use_learnable_positional_encoding)
        self.decoder = Full_Decoder(decoder_language_vocab, context_length, embed_dim, num_layers, num_heads, dropout_rate, use_learnable_positional_encoding, pad_id)
    
    def forward(self, encoder_tokens, decoder_tokens):
        encoder_output = self.encoder(encoder_tokens)
        result = self.decoder(decoder_tokens, encoder_output, encoder_tokens)
        return result

