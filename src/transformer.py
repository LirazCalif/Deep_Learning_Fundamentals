import torch
import torch.nn as nn
import math

def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0]

    values, attention = None, None
    # decide if we have multi-head
    single_head = (q.dim() == 3)
    if single_head:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
    head_size = q.shape[1]

    # windows length
    half_w = window_size // 2
    w_len = window_size + 1

    # pad k and v for borders
    k_pad = torch.zeros(batch_size, head_size, seq_len+window_size, embed_dim, device=k.device, dtype=k.dtype)
    v_pad = torch.zeros(batch_size, head_size, seq_len+window_size, embed_dim, device=v.device, dtype=v.dtype)

    k_pad[:, :, half_w : half_w+seq_len, :] = k
    v_pad[:, :, half_w : half_w+seq_len, :] = v

    # extract sliding windows
    k_win = k_pad.unfold(dimension=2, size=w_len, step=1)
    v_win = v_pad.unfold(dimension=2, size=w_len, step=1)
    k_win = k_win.permute(0, 1, 2, 4, 3).contiguous()
    v_win = v_win.permute(0, 1, 2, 4, 3).contiguous()

    # calc local attention logits
    local_logit = torch.einsum("bhsd,bhswd->bhsw", q, k_win) / math.sqrt(embed_dim)

    # mask out-of-bounds window slots
    idx_s = torch.arange(seq_len, device=q.device).unsqueeze(1)
    idx_w = (torch.arange(w_len, device=q.device).unsqueeze(0) - half_w)
    idx_global = idx_s + idx_w

    in_bounds = (idx_global >= 0) & (idx_global < seq_len)
    local_logit = local_logit.masked_fill(
        ~in_bounds.view(1, 1, seq_len, w_len),
        torch.finfo(local_logit.dtype).min
    )

    # apply padding_mask
    if padding_mask is not None:
        key_is_pad = (padding_mask == 0)
        idx_clamped = idx_global.clamp(0, seq_len - 1)
        idx_flat = idx_clamped.reshape(1, -1).expand(batch_size, -1)

        key_is_pad_win = key_is_pad.gather(1, idx_flat).view(batch_size, seq_len, w_len)

        local_logit = local_logit.masked_fill(
            key_is_pad_win.view(batch_size, 1, seq_len, w_len),
            torch.finfo(local_logit.dtype).min
        )

        query_is_pad = (padding_mask == 0).view(batch_size, 1, seq_len, 1)
        local_logit = local_logit.masked_fill(query_is_pad, 0.0)

    # apply softmax
    attn_local = torch.softmax(local_logit, dim=-1)
    if padding_mask is not None:
        attn_local = attn_local.masked_fill(query_is_pad, 0.0)

    # values in thw windows
    values = torch.einsum("bhsw,bhswd->bhsd", attn_local, v_win)

    # full attention
    attention = torch.zeros(batch_size, head_size, seq_len, seq_len,
                            device=q.device, dtype=attn_local.dtype)

    attn_local = attn_local.masked_fill(~in_bounds.view(1, 1, seq_len, w_len), 0.0)

    idx_scatter = idx_global.clamp(0, seq_len - 1)
    idx_scatter = idx_scatter.view(1, 1, seq_len, w_len).expand(batch_size, head_size, -1, -1)

    attention.scatter_add_(dim=-1, index=idx_scatter, src=attn_local)

    # squeeze if single head
    if single_head:
        values = values.squeeze(1)
        attention = attention.squeeze(1)
    # ======================

    return values, attention

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # call the sliding window attention function you implemented
        values, attention = sliding_window_attention(
            q=q, k=k, v=v,
            window_size=self.window_size,
            padding_mask=padding_mask
        )
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''
        # multi head attention
        multi_head_output = self.self_attn(x, padding_mask)
        # add & norm
        x = self.norm1(x + self.dropout(multi_head_output))
        # feed forward
        feed_forward_output = self.feed_forward(x)
        # add & norm
        x = self.norm2(x + self.dropout(feed_forward_output))
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None
        # Embed tokens + positional encoding
        x = self.encoder_embedding(sentence)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)

        # Classification
        cls_repr = x[:, 0, :]
        output = self.classification_mlp(cls_repr)
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    