import numpy as np
import torch
from torch import nn, Tensor
import pickle, os
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        #Modified to be batch first
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class Transformer(nn.Module):

    #TODO: Source should not have a mask, but target should have a mask. What about padding mask? That might help
    #   improve training by ignoring the padding in the adjacency matrices?
    #TODO: Fix dynamic mask generation for sequences of variable length

    model_id = "Transformer"
    
    def __init__(self, prob_embed, src_pad_token, tgt_pad_token, d_model = 512, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6,
                 dim_feedforward = 2048, dropout = 0.1, activation = 'relu', custom_encoder = None,
                 custom_decoder = None, include_pos = False, target_shape = 50, source_shape = 957,
                 layer_norm_eps = 1e-05, batch_first = True, norm_first = False, add_stop = False, 
                 device = None, dtype = torch.float):
        r"""Most parameters are standard for the PyTorch transformer class. A few specific ones that have been added:
        
        prob_embed: If set to True, then the substructure probabilities are embedded directly using the ProbabilityEmbedding() module when processing the 
            source information. If set to False, then the substructure indices are embedded using the nn.Embedding() module. This boolean must be specified each time.
        src_pad_token: Token for padding out the length of the input sequence
        tgt_pad_token: Token for padding out the length of the target sequence
        include_pos (bool): If positional encoding should be added. Defaults to False (no positional encoding)
        target_shape (int): The size of the molecular graph along one dimension. Expect target sequences
            (adjacency matrices) of shape (target_shape, target_shape)
            In the case of processing SMILES strings, can be thought of as the minimum number of indices
            needed to specify the tgt sequence alphabet (including padding and start tokens).
        source_shape (int): The length of the substructure array, defaults to 957 for the substructure array. 
            Needed to initialize the embedding shape. Can also think of this as the minimum number of indices 
            needed to specify the src sequence (including padding tokens). 
        batch_first is set to be default True, more intuitive to reason about dimensionality if batching 
            dimension is first
        add_stop: Whether a stop token is used during training. Affects the dimension of the target embedding
        """
        super().__init__()
        #Store some information
        self.prob_embed = prob_embed
        self.tgt_shape = target_shape
        self.src_shape = source_shape
        self.include_pos = include_pos
        self.d_model = d_model
        self.dtype = dtype
        self.add_stop = add_stop
        self.device = device
        self.src_pad_token = src_pad_token
        self.tgt_pad_token = tgt_pad_token
        # assert(dropout == 0.5)
        self.transformer = nn.Transformer(d_model = d_model, nhead = nhead, num_encoder_layers = num_encoder_layers,
                                          num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward,
                                          dropout = dropout, activation = activation, custom_encoder = custom_encoder,
                                          custom_decoder = custom_decoder, layer_norm_eps = layer_norm_eps, 
                                          batch_first = batch_first, norm_first = norm_first, device = device, 
                                          dtype = self.dtype)
        
        if self.prob_embed:
            self.src_embed = ProbabilityEmbedding(self.d_model)
        elif not self.prob_embed:
            assert(src_pad_token == 0)
            self.src_embed = nn.Embedding(source_shape + 1, d_model, padding_idx = src_pad_token, dtype = self.dtype, device = self.device)
            assert(self.src_embed.padding_idx == 0)
        #For passing in adjacency matrices
        # self.tgt_embed = nn.Linear(target_shape, d_model, dtype = self.dtype, device = self.device)
        #For passing in smiles index sequences
        #assert(tgt_pad_token in [20, 22, 23])
        if not self.add_stop:
            self.tgt_embed = nn.Embedding(target_shape + 2, d_model, padding_idx = tgt_pad_token, dtype = self.dtype, device = self.device)
        else:
            #Additional +1 for start, pad, and stop tokens
            self.tgt_embed = nn.Embedding(target_shape + 3, d_model, padding_idx = tgt_pad_token, dtype = self.dtype, device = self.device)
        #assert(self.tgt_embed.padding_idx in [20, 22, 23])
        # assert(tgt_pad_token in [20, 22, 23, 33])
        # self.tgt_embed = nn.Embedding(target_shape + 2, d_model, padding_idx = tgt_pad_token, dtype = self.dtype, device = self.device)
        # assert(self.tgt_embed.padding_idx in [20, 22, 23, 33])
        print("Embedding check passed")
        if include_pos:
            print("Including position encoding")
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = None
        if not self.add_stop:
            self.out = nn.Linear(d_model, target_shape + 2, dtype = self.dtype, device=self.device)
        else:
            self.out = nn.Linear(d_model, target_shape + 3, dtype = self.dtype, device=self.device)

    def _get_tgt_mask(self, size):
        #Generate a mask for the target to preserve autoregressive property. Note that the mask is 
        #   Additive for the PyTorch transformer
        mask = torch.tril(torch.ones(size, size, dtype = self.dtype, device = self.device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    
    def _get_tgt_key_pad_mask(self, sizes, tgt_size):
        #Generate a (N, T) matrix which tells which elements of the sequence
        #   to be ignored in the attention. 
        #TODO: Come back and remove this loop somehow
        mask = torch.zeros(len(sizes), tgt_size)
        for i in range(len(sizes)):
            mask[i][:sizes[i]] = False
            mask[i][sizes[i]:] = True
        mask = mask.to(self.device)
        return mask

    def forward(self, src, tgt, sizes = None):
        #We need the sizes here (the actual number of elements for each adj matrix)
        #Along the sequence length dimension, dim = 1 (N, T, E)
        #   This mask is combined with the attention weights
        if sizes is not None:
            assert(len(sizes) == src.shape[0] == tgt.shape[0])
        #TODO: Fix device mismatches throughout!
        #   Tensor.to(None) does not change the tensor device. Since 
        #   self.device is None, need to send tgt_mask to src.device. There is a 
        #   cleaner way of handling this...
        tgt_mask = self._get_tgt_mask(tgt.size(1)).to(src.device)
        #The source mask masks out padding in the input sequence so it does not 
        #   affect the attention mechanism. We want to mask out the 0 elements, 
        #   so we have True where to ignore (i.e., where the sequence equals 0)
        assert(self.src_pad_token == 0)
        #assert(self.tgt_pad_token in [20, 22, 23])
        # assert(self.tgt_pad_token in [20, 22, 23, 33])
        #Only use the source key pad mask if we are not using probability embedding (indices instead)
        if self.prob_embed:
            src_key_pad_mask = None
        else:
            src_key_pad_mask = (src == self.src_pad_token).bool().to(self.device)

        tgt_key_pad_mask = (tgt == self.tgt_pad_token).bool().to(self.device)
        # if sizes is None:
        #     tgt_key_pad_mask = None
        # else:
        #     tgt_key_pad_mask = self._get_tgt_key_pad_mask(sizes, tgt.size(1)).bool()
        src = self.src_embed(src) * math.sqrt(self.d_model)
        #Be sure to convert the adjacency matrix to probabilities rather than hard 
        #   1 and 0. 
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        #Conditional positional encoding
        # print(f"src device", src.device)
        # print(f"tgt device", tgt.device)
        # print(f"tgt_mask device", tgt_mask.device)
        # print(f"tgt_key_pad_mask device", tgt_key_pad_mask.device)
        # print(f"src_key_pad_mask device", src_key_pad_mask.device)
        if self.include_pos:
            src = self.pos_encoder(src)
            tgt = self.pos_encoder(tgt)
        transformer_out = self.transformer(src, tgt, 
                                           tgt_mask = tgt_mask, 
                                           tgt_key_padding_mask = tgt_key_pad_mask,
                                           src_key_padding_mask = src_key_pad_mask)
        #Removed softmax application, the cross entropy loss expects unnormalized logits
        out = self.out(transformer_out)
        return out