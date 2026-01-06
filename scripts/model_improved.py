# This model implements a Transformer architecture with specialized token embeddings
# that extract color, shape, and quantity features from card tokens.
import torch
import torch.nn as nn
import math

# In the WCST task, each card is defined by multiple categorical features—color, shape, and quantity—so representing them as a single token index allows the model 
# to process cards efficiently while preserving all feature information. 
# By exploring this type of token embedding, we can examine whether the model can internally disentangle and utilize these features for rule-based reasoning. 
# This approach also allows us to probe how different components, like positional encodings or attention mechanisms, interact with the structured, multi-feature input.

class TokenEmbeddingWithFeatures(nn.Module):

    def __init__(self, vocab_size, d_model, dropout):
        super(TokenEmbeddingWithFeatures, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # split d_model into three parts, distributing any remainder to the last part
        base = d_model // 3
        rem = d_model - base * 3
        dims = [base, base, base + rem]
        
        self.color_embed = nn.Embedding(4, dims[0])   # 4 colors
        self.shape_embed = nn.Embedding(4, dims[1])   # 4 shapes
        self.quantity_embed = nn.Embedding(4, dims[2])  # 4 quantities
        
        # For special tokens (64-69), there are 6 special tokens (64..69)
        self.special_embed = nn.Embedding(6, d_model)  # tokens 64-69 inclusive
        
        self.scaling = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(dropout)
    
    # Now extract color, shape, quantity from card token index
    def extract_features(self, token_idx):
        # Each token index is a single number that encodes three card features—color, shape, and quantity—each with 4 possible values. 
        # We can think of the token index like a three-digit base-4 number: the first “digit” gives the color, the second gives the shape, and the third gives the quantity. 
        # We have used integer division and modulo to extract each feature from this compact representation.
        color = token_idx // 16      # We extract the first attribute  0-3
        shape = (token_idx % 16) // 4  # 0-3
        quantity = token_idx % 4      # 0-3
        return color, shape, quantity
    
    def positional_encoding(self, input_x):
        batch_size, seq_length, _ = input_x.shape
        pe = torch.empty(seq_length, self.d_model, device=input_x.device, dtype=input_x.dtype)
        
        for p in range(seq_length):
            for i in range(0, self.d_model, 2):
                pe[p, i] = math.sin(p / 10000 ** (i / self.d_model))
                if i + 1 < self.d_model:
                    pe[p, i+1] = math.cos(p / 10000 ** (i / self.d_model))
        
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return pe
    
    def forward(self, input_x):
        batch_size, seq_length = input_x.shape
        embeddings = torch.zeros(batch_size, seq_length, self.d_model, 
                                device=input_x.device, dtype=torch.float32)
        
        for b in range(batch_size):
            for s in range(seq_length):
                token = int(input_x[b, s].item())
                
                if token < 64:  # Card token - use feature extraction
                    color, shape, quantity = self.extract_features(token)
                    
                    color_emb = self.color_embed(torch.tensor(color, device=input_x.device, dtype=torch.long))
                    shape_emb = self.shape_embed(torch.tensor(shape, device=input_x.device, dtype=torch.long))
                    quantity_emb = self.quantity_embed(torch.tensor(quantity, device=input_x.device, dtype=torch.long))
                    
                    embeddings[b, s, :] = torch.cat([color_emb, shape_emb, quantity_emb])
                else:  # Special token (category, SEP, EOS)
                    special_idx = token - 64
                    embeddings[b, s, :] = self.special_embed(torch.tensor(special_idx, device=input_x.device, dtype=torch.long))
        
        
        pos_enc = self.positional_encoding(embeddings)
        embeddings = embeddings * self.scaling
        embeddings = embeddings + pos_enc
        embeddings = self.dropout(embeddings)
        
        return embeddings


# All the other classes (MH_Attention, FFNetwork, Encoder, TransformerWithFeatures) remain unchanged from the baseline model
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
        self.out = nn.Linear(d_model, d_model, bias =True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        
        Q_proj = self.Q(query)
        K_proj = self.K(key)
        V_proj = self.V(value)
    
        if mask is not None: 
            mask = mask.unsqueeze(0).unsqueeze(1)  
        
        head_chunks_q = torch.split(Q_proj, self.d_head, dim=-1)
        query = torch.stack(head_chunks_q, dim=2)
        query = query.transpose(1,2)

        head_chunks_k = torch.split(K_proj, self.d_head, dim=-1)
        key = torch.stack(head_chunks_k, dim=2)
        key = key.transpose(1,2)

        head_chunks_v = torch.split(V_proj, self.d_head, dim=-1)
        value = torch.stack(head_chunks_v, dim=2)
        value = value.transpose(1,2)

        att_scores = torch.matmul(query, key.transpose(-2, -1))
        att_scores = att_scores / math.sqrt(self.d_head)

        if mask is not None: 
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        
        att_weights = torch.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights) 

        att_output = torch.matmul(att_weights, value)
        att_output = att_output.transpose(1,2)
        att_output = torch.cat(torch.unbind(att_output, dim=2), dim=-1)

        output = self.out(att_output)

        return output, att_weights

class FFNetwork(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout, bias=True):
        super(FFNetwork, self).__init__()

        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.dropout = dropout

        self.FF1 = nn.Linear(self.d_model, self.d_feedforward, bias=True)
        self.FF2 = nn.Linear(self.d_feedforward, self.d_model, bias=True)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.FF1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.FF2(x)
        return x
    
        
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout = 0.1, bias=True):
        super(Encoder, self).__init__()

        self.att_layer = MH_Attention(d_model, num_heads, dropout, bias)
        self.ffn_layer = FFNetwork(d_model, d_ff, dropout, bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x, mask):
        
        #Post-Layernorm implemented !!!
        attention_output, attention_weights = self.att_layer(x, x, x, mask) # Self-attention where we pass q,k,v, and mask=true/false depending on if we want masked attention
        x = self.norm1(x + self.dropout(attention_output))

        ffn_output = self.ffn_layer(x) # Feed Forward Network output
        x = self.norm2(x + self.dropout(ffn_output))

        return x , attention_weights

# Transformer model with multi-feature token embeddings
class TransformerWithFeatures(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, num_heads, dropout):
        super(TransformerWithFeatures, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.embedding = TokenEmbeddingWithFeatures(vocab_size, d_model, dropout)
        
        self.encoders = nn.ModuleList()
        for _ in range(num_layers):
            # use the local Encoder class
            self.encoders.append(Encoder(d_model, d_ff, num_heads, dropout))

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask, return_attention=False):
        x = self.embedding(x)

        attention_weights = []
        for encoder in self.encoders:
            x, att_weights = encoder(x, mask)
            if return_attention:
                attention_weights.append(att_weights)

        x = self.output_layer(x)

        if return_attention:
            return x, attention_weights
        else:
            return x