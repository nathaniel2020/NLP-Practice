#!/usr/bin/env python
# coding: utf-8

# In[64]:


import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


# ## 1. 准备数据
# 

# In[34]:


# S: start
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# vocabulary dictionary
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5} # Source language
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8} # target language
idx_to_word = {i: word for i, word in enumerate(tgt_vocab)} # to translate the indexes sequence to words sequence
tgt_vocab_size = len(tgt_vocab)

enc_max_len = 5 # the max sequence length of enc_input
dec_max_len = 6 # the max sequence length of dec_input and dec_output


# In[ ]:


def make_dataset(sentences):
    '''
    transform the word sentences into indexes tensor
    :param sentences: list of list(have three elements: enc_input, dec_input, dec_output) of string
    :return: Three torch.LongTensor elements: enc_inputs, dec_inputs, dec_outputs
            size: (batch_size, max_len(enc or dec))
    '''
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [src_vocab[word] for word in sentences[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentences[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentences[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


# In[36]:


enc_inputs, dec_inputs, dec_outputs = make_dataset(sentences)


# In[37]:


class Dataset(Data.Dataset):
    '''
    construct our own dataset
    '''
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(Dataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# In[38]:


data_loader = Data.DataLoader(Dataset(enc_inputs, dec_inputs, dec_outputs))


# In[39]:



# ## 2. 定义模型

# ### 2.1 Parameter

# In[66]:


d_model = 512
d_k = d_v = 64 # (d_model / n_heads) the dimension of Q, K must be equal, and V does not have limit
n_heads = 8 # multi head attention
d_ff = 2048 # the dimension of feed forward
n_layers = 6 # number of encoder layer and decoder layer
lr=1e-3 # learning rate
momentum=0.99 # parameters of SGD
EPOCH = 10


# ### 2.2 Positional Encoding

# In[67]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''

        :param d_model: the dimension of one word embedding
        :param dropout:
        :param max_len: the maximum length of input data, just need bigger than the batch_size
        '''
        super(PositionalEncoding, self).__init__()

        pe = torch.ones(max_len, d_model)
        pe = pe * torch.arange(max_len).reshape(-1, 1) # to make every row's value is there index
        pe = pe / torch.pow(10000, torch.arange(d_model) * 2 / d_model)
        pe[:, 0::2] = torch.sin(pe)[:, 0::2] # the dimension index is even
        pe[:, 1::2] = torch.cos(pe)[:, 1::2] # odd
        # pe: (max_len, 1, d_model)
        pe = pe.unsqueeze(1) # because the input x has three dimension (batch_size, seq_len, d_model)
        self.register_buffer('pe', pe) # register the tensor into buffer that it will not be update by optimizer

    def forward(self, x):
        '''
        seq_len is equal enc_max_len or dec_max_len
        :param x: (batch_size, seq_len, d_model) the word embedding of a batch
        :return: (batch_size, seq_len, d_model) which add the positional information
        '''
        x = x + self.pe[:x.shape[0],: ,:]
        return x


# In[68]:


# ## 2.3 Padding Mask
# To deal with the non-fixed length sequence
# We need use padding to fill the short sequence
# and the padding mark does not supply information
# so we use matrix to mask this padding
# and after softmax the corresponding probability is 0
# 
# 不定长文本需要截断或者填充，而填充标记不提供任何有用的信息，
# 所以使用矩阵来把这部分遮盖起来，使得softmax计算后其对应的概率为0

# In[69]:


def get_padding_mask(seq_q, seq_k):
    '''
    seq_q is the query
    and we want to know the influence to seq_q of each element in sek_k
    so is the element in seq_k is padding mark, it can not supply information
    thus we need to change the score with a minim value before softmax
    This function is just a mark that assign the position
    which the value is True should be modify with a minim value

    seq_q and seq_k is the raw input(haven't through embedding layer)
    :param seq_q: (batch_size, q_seq_len)
    :param seq_k: (batch_size, k_seq_len) k_seq_len may be different with q_seq_len
    :return: (batch_size, q_seq_len, k_seq_len)
    '''
    batch_size, q_len = seq_q.size()
    _, k_len = seq_k.size()
    mask = seq_k.data.eq(0)
    # expand do not allocate new memory, it just creates a new view
    return mask.expand(batch_size, q_len, k_len)


# In[70]:



# ## 2.4 Sequence Mask
# 
# Prevent disclosure of information

# In[45]:


def get_sequence_mask(seq):
    '''
    only used in decoder, the position which value equal 1 should be masked
    :param seq: (batch_size, seq_len=dec_max_len)
    :return: (batch_size, seq_len=dec_max_len, seq_len=dec_max_len) a lower triangular matrix
    '''
    mask_size = seq.size()[0], seq.size()[1], seq.size()[1]
    sequence_mask = torch.triu(torch.ones(mask_size), diagonal=1) # Upper triangular matrix
    return sequence_mask


# In[46]:




# ## 2.5 Scaled Dot Product Attention

# In[47]:


def scaled_dot_product_attention(Q, K, V, mask):
    '''
    in transformer, it has thress attention operation
    two of them are self-attention that q_seq_len equal to k_seq_len
    one of them is attention(the output of encoder as Q and V, the output of front part decoder as Q)
    :param Q: torch.tensor (batch_size, n_heads, q_seq_len, d_k)
    :param K: torch.tensor (batch_size, n_heads, k_seq_len, d_k)
    :param V: torch.tensor (batch_size, n_heads, v_seq_len=k_seq_len, d_v)
    :param mask: torch.tensor (batch_size, n_heads, q_seq_len, k_seq_len)
    :return: torch.tensor (batch_size, n_heads, q_seq_len, d_v)
            torch.tensor (batch_size, n_heads. q_seq_len, k_seq_len)
    '''
    softmax = nn.Softmax(dim=-1) # we need do softmax in d_k's dimension(the row and is the last dimension, also can use 3 int this proejct)

    sources = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(d_k) # exchange the last dimension of K
    sources.masked_fill(mask, -1e9) # mask operation
    atten_source = softmax(sources)
    return torch.matmul(atten_source, V), atten_source


# ## 2.6 MultiHeadAttention

# In[48]:


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, mask):
        '''

        :param Q: torch.tensor (batch_size, q_seq_len, d_model)
        :param K: torch.tensor (batch_size, k_seq_len, d_model)
        :param V: torch.tensor (batch_size, v_seq_len=k_seq_len, d_model)
        :param mask: torch.tensor (batch_size, q_seq_len, k_seq_len)
        :return: context: (batch_size, q_seq_len, d_model)
                atten: (batch_size, n_heads, q_seq_len, k_seq_len)
        '''
        batch_size, q_seq_len, k_seq_len = mask.size()
        residual = Q # residual connection
        # through the liner layer, do attention in a small(d_k, d_v dimension) projection space
        # self.W_Q(Q): (batch_size, q_seq_len, n_heads*d_model)
        Q = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2) # Q: (batch_size, n_heads, q_seq_len, d_k)
        K = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        V = self.W_V(V).view(batch_size, -1, n_heads, d_k).transpose(1,2)

        mask = mask.unsqueeze(1).expand(batch_size, n_heads, q_seq_len, k_seq_len)
        # context: (batch_size, n_heads, q_seq_len, d_v)
        # atten: (batch_size, n_heads, q_seq_len, k_seq_len)
        context, atten = scaled_dot_product_attention(Q, K, V, mask)

        # project the vector to the original dimension size(d_model)
        # if do not use contiguous(), also can use reshape function to replace view
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads * d_v)
        context = self.fc(context) # (batch_size, q_seq_len, d_model)
        return nn.LayerNorm(d_model)(context + residual), atten # residual connection and layernorm


# ## 2.7 FeedForward Layer
# 

# In[49]:


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, input):
        '''

        :param input: torch.tensor (batch_size, q_seq_len, d_model) the output of multi head attention
        :return: (batch_size, q_seq_len, d_model)
        '''
        residual = input
        output = self.fc(input) # (batch_size, q_seq_len, d_model)
        return nn.LayerNorm(d_model)(output + residual)


# ## 2.8 Encoder Layer

# In[50]:


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # attention in encoder is self attention
        self.enc_multi_self_atten = MultiHeadAttention()
        self.feed_forward = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, mask):
        '''
        the mask is self attention's mask, obtain from the get_padding_mask() function
        :param enc_inputs: torch.tensor (batch_size, enc_max_len, d_model)
        :param mask: torch.tensor (batch_size, q_seq_len=enc_max_len, k_seq_len=enc_max_len)
        :return:
        '''
        # context: (batch_size, q_seq_len=enc_max_len, d_model)
        # atten: (batch_size, n_heads, q_seq_len=enc_max_len, k_seq_len=enc_max_len)
        context, atten = self.enc_multi_self_atten(enc_inputs, enc_inputs, enc_inputs, mask)
        context = self.feed_forward(context) # (batch_size, q_seq_len=enc_max_len, d_model)
        return context, atten


# ## 2.9 Encoder

# In[57]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # the size of corpora, the dimension of the embedding
        self.emb = nn.Embedding(src_vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''

        :param enc_inputs: torch.tensor (batch_size, enc_max_len)
        :return: enc_outputs: torch.tensor (batch_size, enc_max_len, d_model)
                enc_self_attn: list(length = n_layers) of torch.tensor(size: (batch_size, q_seq_len=enc_max_len, k_seq_len=enc_max_len))
        '''
        enc_outputs = self.emb(enc_inputs) # (batch_size, enc_max_len, d_model)
        enc_outputs = self.pos(enc_outputs) # (batch_size, enc_max_len, d_model)

        enc_self_attn = [] # a list that store the each encoder layer's attention result
        mask = get_padding_mask(enc_inputs, enc_inputs) # encoder should use padding mask
        for encoder_layer in self.encoder_layers:
            # enc_outputs: (batch_size, enc_max_len, d_model)
            # atten: (batch_size, q_seq_len=enc_max_len, k_seq_len=enc_max_len)
            enc_outputs, atten = encoder_layer(enc_outputs, mask)
            enc_self_attn.append(atten)
        return enc_outputs, enc_self_attn


# ## 2.10 Decoder Layer

# In[58]:


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_multi_self_atten = MultiHeadAttention()
        self.dec_multi_enc_atten = MultiHeadAttention()
        self.feed_forward = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_atten_mask, dec_enc_atten_mask):
        '''

        :param dec_inputs: torch.tensor (batch_size, q_seq_len=dec_max_len, d_model)
        :param enc_outputs: torch.tensor (batch_size, k_seq_len=enc_max_len, d_model)
        :param dec_self_atten_mask: torch.tensor (batch_size, q_seq_len=dec_max_len, k_seq_len=dec_max_len)
        :param dec_enc_atten_mask: torch.tensor (batch_size, q_seq_len=dec_max_len, k_seq_len=enc_max_len)
        :return:
        '''
        # dec_outputs: (batch_size, q_seq_len=dec_max_len, d_model)
        # dec_self_atten: (batch_size, q_seq_len=dec_max_len, k_seq_len=dec_max_len)
        dec_outputs, dec_self_atten = self.dec_multi_self_atten(dec_inputs, dec_inputs, dec_inputs, dec_self_atten_mask)
        # dec_outputs: (batch_size, q_seq_len=dec_max_len, d_model)
        # dec_enc_atten: (batch_size, q_seq_len=dec_max_len, k_seq_len=enc_max_len)
        dec_outputs, dec_enc_atten = self.dec_multi_enc_atten(dec_outputs, enc_outputs, enc_outputs, dec_enc_atten_mask)
        dec_outputs = self.feed_forward(dec_outputs) # (batch_size, q_seq_len=dec_max_len, d_model)
        return dec_outputs, dec_self_atten, dec_enc_atten


# ## 2.11 Decoder

# In[59]:


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''

        :param dec_inputs: (batch_size, dec_max_len)
        :param enc_inputs: (batch_size, enc_max_len)
        :param enc_outputs: (batch_size, enc_max_len, d_model)
        :return: torch.tensor (batch_size, dec_max_len, d_model)
                list(length=n_layers) of torch.tensor((batch_size, dec_max_len, dec_max_len))
                list(length=n_layers) of torch.tensor((batch_size, dec_max_len, enc_max_len))
        '''

        dec_outputs = self.emb(dec_inputs)
        dec_outputs = self.pos(dec_outputs)

        # decoder self attention should prevent disclosure of information
        # thus we need combine the two mask together
        dec_self_atten_seq_mask = get_sequence_mask(dec_inputs) # 1 or 0 (batch_size, dec_max_len, dec_max_len)
        dec_self_atten_pad_mask = get_padding_mask(dec_inputs, dec_inputs) # True or False (batch_size, dec_max_len, dec_max_len)
        dec_self_atten_mask = torch.gt((dec_self_atten_pad_mask + dec_self_atten_seq_mask), 0) # value > 0 is True
        # dec_outputs as the query
        dec_enc_atten_mask = get_padding_mask(dec_inputs, enc_inputs) # (batch_size, dec_max_len, enc_max_len)

        dec_self_attns, dec_enc_attns = [], []
        for decoder_layer in self.decoder_layers:
            dec_outputs, dec_self_atten, dec_enc_atten = decoder_layer(dec_outputs, enc_outputs, dec_self_atten_mask, dec_enc_atten_mask)
            dec_self_attns.append(dec_self_atten)
            dec_enc_attns.append(dec_enc_atten)

        return dec_outputs, dec_self_attns, dec_enc_attns


# ## 2.12 Transformer

# In[62]:


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # use CrossEntropyLoss so in the last layer we don't need to do softamx
        self.liner = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        '''

        :param enc_inputs: torch.tensor (batch_size, enc_max_len)
        :param dec_inputs: torch.tensor (batch_size, dec_max_len)
        :return: outputs: torch.tensor (batch_size * dec_max_len, tgt_vocab_size)
                enc_self_attn: list(length = n_layers) of torch.tensor(size: (batch_size, q_seq_len=enc_max_len, k_seq_len=enc_max_len))
                dec_self_atten: list(length=n_layers) of torch.tensor((batch_size, dec_max_len, dec_max_len))
                dec_enc_atten: list(length=n_layers) of torch.tensor((batch_size, dec_max_len, enc_max_len))
        '''
        enc_outputs, enc_self_attn = self.encoder(enc_inputs)
        dec_outputs, dec_self_atten, dec_enc_atten = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        outputs = self.liner(dec_outputs) # (batch_size, dec_max_len, tgt_vocab_size)
        outputs = outputs.view(-1, outputs.size(-1)) # (batch_size * dec_max_len, tgt_vocab_size)
        return outputs, enc_self_attn, dec_self_atten, dec_enc_atten


# ## 2.13 Training

# In[61]:


model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# In[76]:


for epoch in range(EPOCH):
    for enc_inputs, dec_inputs, dec_outputs in data_loader:
        outputs, *_ = model(enc_inputs, dec_inputs)
        print(outputs.size(), dec_outputs.size())
        break
    break


# In[ ]:




