import torch
import torch.nn as nn 

class SelfAttention(nn.Module):
    """
    Self-attention mechanism.

    This is the multi-head attention from the paper "Attention is all you need", the core component of the transformer model.
    It is used to compute the attention weights for the input embeddings.
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # why we need to initialize the super class?
        # because we want to inherit the properties of the nn.Module class
        # so that we can use the forward method

        self.embed_size = embed_size # dimension of the input embeddings
        self.heads = heads # number of attention heads
        self.head_dim = embed_size // heads # dimension of each head

        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        # why is this assert necessary?
        # because the input embeddings must be divisible by the number of heads
        # otherwise, the values, keys, and queries will not be of the same dimension

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) # linear layer to project the values
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # linear layer to project the keys
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) # linear layer to project the queries
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) # linear layer to project the output

    def forward(self, values, keys, query, mask):
        """
        forward pass of the self-attention mechanism, the input is the embeddings of the input sequence. 
        values, keys, query are the embeddings of the input sequence
        mask is the mask for the input sequence
        the output is the attention weights of the input sequence        
        """
        N = query.shape[0]
        # N is the batch size
        # The shape of the query is [batch_size, length, d_tensor]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embeddings into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # what is einsum?
        # it is a function that allows us to perform the dot product of the queries and keys
        # it is a more efficient way to perform the dot product of the queries and keys

        # why we pass "nqhd,nkhd->nhqk", [queries, keys]
        # because we want to perform the dot product of the queries and keys
        # and the result is a tensor with shape (N, heads, query_len, key_len)
        # nqhd is the shape of the queries
        # nkhd is the shape of the keys
        # nhqk is the shape of the energy

        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # why we need to mask the energy?
        # because we don't want to attend to the padding tokens 
        # we will mask the energy at the position where the mask is 0

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # why we need to divide the energy by the embed_size ** (1/2)?
        # because we want to scale the energy so that the softmax is not too small or too large

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after the einsum, the shape is: (N, query_len, heads, head_dim) then flatten last two dim

        out = self.fc_out(out)
        # out shape: (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    """Transformer block is used to encode the input sequence into a sequence of embeddings"""

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # why we need to pass value, key, query, mask to the attention?
        # because we want to compute the attention weights for the input sequence
        # value, key, query are the embeddings of the input sequence
        # mask is the mask for the input sequence
        attention = self.attention(value, key, query, mask)

        x=self.dropout(self. norm1(attention+query))
        forward=self.feed_forward(x)
        out=self.dropout(self.norm2(x+forward))
        # why we need to add the query to the attention and the feed_forward?
        # because we want to add the attention and the feed_forward to the query
        # so that we can get the output of the transformer block

        return out

class Encoder(nn.Module):
    """Encoder is used to encode the input sequence into a sequence of embeddings"""

    def __init__(self, src_vocab_size, embed_size, heads, dropout, forward_expansion, num_layers, max_length, device):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # why we need to use the embedding layer?
        # because we want to convert the input sequence to a sequence of embeddings
        # the input sequence is a sequence of integers, and the embedding layer converts each integer to a vector
        # the embedding layer is a lookup table, and the input sequence is the index of the lookup table    

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layers)]
        )
        # why we need to use the ModuleList?
        # because we want to store the transformer blocks in a list
        # so that we can iterate over the list and apply the transformer blocks to the input sequence

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # why we need to expand the positions?
        # because we want to create a sequence of positions for the input sequence
        # the positions are the positions of the input sequence
        # the positions are a sequence of integers, and the embedding layer converts each integer to a vector

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # why we need to add the word embedding and the position embedding?
        # because we want to add the word embedding and the position embedding to the input sequence
        # so that we can get the input sequence with the word embedding and the position embedding

        for layer in self.layers:
            out = layer(out, out, out, mask)
        # why we need to pass out, out, out, mask to the transformer block?
        # because we want to apply the transformer block to the input sequence
        # out, out, out are the embeddings of the input sequence
        # mask is the mask for the input sequence   
        return out

class DecoderBlock(nn.Module):
    """Decoder block is the core component of the decoder and it is used to decode the encoded sequence into a sequence of embeddings"""

    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(DecoderBlock, self).__init__()
        self.device = device

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        # why we need to pass value, key, query, src_mask to the transformer block?
        # because we want to apply the transformer block to the input sequence
        # value, key, query are the embeddings of the input sequence
        # src_mask is the mask for the input sequence
        # tgt_mask is the mask for the target sequence
        return out

class Decoder(nn.Module):
    """Decoder is used to decode the encoded sequence into a sequence of embeddings"""

    def __init__(self, tgt_vocab_size, embed_size, heads, dropout, forward_expansion, num_layers, max_length, device):
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, dropout, forward_expansion, device) for _ in range(num_layers)]

        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x)+self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
        # why we need to pass x, enc_out, enc_out, src_mask, tgt_mask to the decoder block?
        # because we want to apply the decoder block to the input sequence
        # x is the embeddings of the input sequence
        # enc_out is the embeddings of the encoded sequence
        # src_mask is the mask for the input sequence
        # tgt_mask is the mask for the target sequence
        out = self.fc_out(x)
        # why we need to pass x to the linear layer?
        # because we want to project the embeddings of the input sequence to the target vocabulary size
        return out

class Transformer(nn.Module):
    """Transformer is the main component of the transformer model. In the class, we will define the encoder, decoder, and the transformer block."""

    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, embed_size=256, heads=8, dropout=0, forward_expansion=4, num_layers=6, max_length=100, device="cuda"):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, heads, dropout, forward_expansion, num_layers, max_length, device)
        self.decoder = Decoder(tgt_vocab_size, embed_size, heads, dropout, forward_expansion, num_layers, max_length, device)
        #we need to pass the src_pad_idx and tgt_pad_idx to the encoder and decoder
        # because we need to mask the padding tokens in the input sequence and target sequence

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        # why we need to unsqueeze the src_mask?
        # because we need to mask the padding tokens in the input sequence
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # (trg_len, trg_len)
        # why we need to create a triangular matrix?
        # because we need to mask the future tokens in the target sequence
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# The transformer is created!


        
        

        
        
        
        
        
        
        
        
        
        
        
        




        
        
        


        



        



  
        
        

        

