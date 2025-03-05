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






        
        
        


        



        



  
        
        

        

