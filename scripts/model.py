# This is the baseline Transformer model implementation.
# I took inspiration from https://pi-tau.github.io/posts/transformer/#transformer and ...
# https://huggingface.co/datasets/bird-of-paradise/transformer-from-scratch-tutorial/blob/main/Transformer_Implementation_Tutorial.ipynb
# These two blog sites helped me understand the architecture and implementation details, the difference however is that I only used Encoder blocks, 
# since we are doing next-token prediction only, not sequence-to-sequence tasks, so there is no decoder here.

import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout):
        super(TokenEmbedding, self).__init__()

        self.d_model = d_model
        #self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        #self.proj = nn.Linear(self.embed_size, self.d_model)

        self.scaling = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(dropout)

    
    def positional_encoding(self,input_x):
        batch_size, seq_length, _ = input_x.shape

        pe = torch.empty(seq_length, self.d_model, device=input_x.device, dtype=input_x.dtype) # create empty tensor of seq length x d_model
        
        # I broke down the calculations for clarity, hence why i used math lib and not torch, and it is more intuitive this way for me, however for efficiency this can be vectorized
        for p in range(seq_length):
            for i in range(0, self.d_model, 2):
                
                pe[p, i] = math.sin(p / 10000 ** (i / self.d_model))

                if i + 1 < self.d_model:
                    pe[p, i+1] = math.cos(p / 10000 ** (i / self.d_model))

        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)

        return pe
    
    def forward(self, input_x):


        input_x = self.token_embedding(input_x)* self.scaling
        #input_x = self.proj(input_x) * self.scaling

        pos_enc = self.positional_encoding(input_x)
        
        
        input_x = input_x + pos_enc 
        input_x = self.dropout(input_x)

        return input_x



class MH_Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, bias):
        super(MH_Attention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias

        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.Q = nn.Linear(d_model, d_model, bias =False)
        self.K = nn.Linear(d_model, d_model, bias =False)
        self.V = nn.Linear(d_model, d_model, bias =False)
        self.out = nn.Linear(d_model, d_model, bias =True) # bias is true for output layer

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        
        # Project Q, K, V
        Q_proj = self.Q(query)
        K_proj = self.K(key)
        V_proj = self.V(value)
    
        if mask is not None: 
            mask = mask.unsqueeze(0).unsqueeze(1)  
        
        # Linear projections and split into heads, i broke it down as it is more intuitive this way for me
        # .... instead of using .view() and/or .permute(), however using those in-built functions would be more efficient and less prone to error
        head_chunks_q = torch.split(Q_proj, self.d_head, dim=-1)
        query = torch.stack(head_chunks_q, dim=2)
        query = query.transpose(1,2)

        head_chunks_k = torch.split(K_proj, self.d_head, dim=-1)
        key = torch.stack(head_chunks_k, dim=2)
        key = key.transpose(1,2)

        head_chunks_v = torch.split(V_proj, self.d_head, dim=-1)
        value = torch.stack(head_chunks_v, dim=2)
        value = value.transpose(1,2)

        # Attention Scores -> Q * K^T / sqrt(d_head)
        att_scores = torch.matmul(query, key.transpose(-2, -1))
        att_scores = att_scores / math.sqrt(self.d_head) # SCALE BY DIM IS V IMPORTANT !!!

        # For masked attention, fill with -infinity....
        if mask is not None: 
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        
        # Attention Weights -> softmax
        att_weights = torch.softmax(att_scores, dim=-1) # Attention weights = softmax(scores)
        att_weights = self.dropout(att_weights) 

        # Concatenate heads
        att_output = torch.matmul(att_weights, value) # Attention output = weights * V
        att_output = att_output.transpose(1,2)
        att_output = torch.cat(torch.unbind(att_output, dim=2), dim=-1)

        # Final output 
        output = self.out(att_output)

        return output, att_weights

class FFNetwork(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout, bias=True):
        super(FFNetwork, self).__init__()

        # FFN(X) = W2​(ReLU(W1*​x + b1​))+b2​

        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.dropout = dropout

        self.FF1 = nn.Linear(self.d_model, self.d_feedforward, bias=True) # W1*x + b1
        self.FF2 = nn.Linear(self.d_feedforward, self.d_model, bias=True) # W2* (ReLU(...)) + b2

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Two layer MLP with activation and dropout

        x = self.FF1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.FF2(x)

        return x
    
        
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout = 0.1, bias=True):
        super(Encoder, self).__init__()

        # Multi-Head Attention Layer
        self.att_layer = MH_Attention(d_model, num_heads, dropout, bias)

        # Feed Forward Network
        self.ffn_layer = FFNetwork(d_model, d_ff, dropout, bias)

        # Layer normalization + residual connections
        self.norm1 = nn.LayerNorm(d_model) # After attention layer
        self.norm2 = nn.LayerNorm(d_model) # After feed forward layer

        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x, mask):
        
        #Post-Layernorm implemented !!!
        attention_output, attention_weights = self.att_layer(x, x, x, mask) # Self-attention where we pass q,k,v, and mask=true/false depending on if we want masked attention
        x = self.norm1(x + self.dropout(attention_output))

        ffn_output = self.ffn_layer(x) # Feed Forward Network output
        x = self.norm2(x + self.dropout(ffn_output))

        return x , attention_weights # I returenn attention weights for possible analysis later on

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        #self.embed_size = embed_size
        self.d_model = d_model 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.embedding = TokenEmbedding(vocab_size, d_model, dropout)
        
        self.encoders = nn.ModuleList()
        for _ in range(num_layers):
            self.encoders.append(Encoder(d_model, d_ff, num_heads, dropout))

        self.output_layer = nn.Linear(d_model, vocab_size)


    def forward(self, x, mask, return_attention=False):
        x = self.embedding(x)

        attention_weights = []  # Store attention from all layers

        for encoder in self.encoders:
            x, att_weights = encoder(x, mask)
            if return_attention:
                attention_weights.append(att_weights)

        x = self.output_layer(x)

        if return_attention:
            return x, attention_weights
        else:
            return x

