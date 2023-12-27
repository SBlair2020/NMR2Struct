import numpy as np
import torch
from torch import nn, Tensor
import pickle, os
import math
import torch.nn.functional as F
from typing import Optional, Any, Tuple, Callable

class PositionalEncoding(nn.Module):

    """ Positional encoding with option of selecting specific indices to add """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 30000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor, ind: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            ind: Tensor, shape [batch_size, seq_len] or NoneType
        '''
        #Select and expand the PE to be the right shape first
        if ind is not None:
            added_pe = self.pe[torch.arange(1).reshape(-1, 1), ind, :]
            x = x + added_pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Transformer(nn.Module):

    model_id = "Transformer"
    
    def __init__(self, src_embed: nn.Module,
                 src_pad_token: int, tgt_pad_token: int, 
                 src_embedding_function: Callable[[Tensor, nn.Module, int, Optional[nn.Module]], Tuple[Tensor, Optional[Tensor]]],
                 tgt_embedding_function: Callable[[Tensor, nn.Module, int, Optional[nn.Module]], Tuple[Tensor, Optional[Tensor]]],
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu', custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None, include_pos: bool = False, target_shape: int = 50, source_shape: int = 957,
                 layer_norm_eps: float = 1e-05, batch_first: bool = True, norm_first: bool = False, add_stop: bool = False, 
                 device: torch.device = None, dtype: torch.dtype = torch.float):
        #TODO: Update docstring
        r"""Most parameters are standard for the PyTorch transformer class. A few specific ones that have been added:
        
        prob_embed: This is now a string indicating which kind of probability embedding layer to use. The options are as follows:
            'nn.embed': Tokenized embedding for the source, where the tokens are represented as integers ranging from [0, n_substructures). This uses
                PyTorch's built-in nn.Embedding layer.
            'mlp': A one-hidden layer MLP. The middle activation is ReLU() and the final activation is Tanh() to ensure embedding values are within the 
                interval [-1, 1]
            'matrix_scale': A element-wise multiplication between the input vectors and a matrix of weights. The input sequence is expected to be shape 
                (batch_size, seq_len, 1), the weight matrix is shape (seq_len, d_model) and the output is shape (batch_size, seq_len, d_model). This relies on 
                broadcasting.
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

        self.src_embed = src_embed
        self.src_embed_fn = src_embedding_function
        self.tgt_embed_fn = tgt_embedding_function
        self.src_pad_token = src_pad_token
        self.tgt_pad_token = tgt_pad_token

        self.tgt_shape = target_shape
        self.src_shape = source_shape
        self.include_pos = include_pos
        self.d_model = d_model
        self.dtype = dtype
        self.device = device

        self.transformer = nn.Transformer(d_model = d_model, nhead = nhead, num_encoder_layers = num_encoder_layers,
                                          num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward,
                                          dropout = dropout, activation = activation, custom_encoder = custom_encoder,
                                          custom_decoder = custom_decoder, layer_norm_eps = layer_norm_eps, 
                                          batch_first = batch_first, norm_first = norm_first, device = device, 
                                          dtype = self.dtype)
        
        #TODO: Cleaner way of initializing embeddings?
        # if self.prob_embed == 'mlp':
        #     self.src_embed = ProbabilityEmbedding(self.d_model)
        # elif self.prob_embed == 'matrix_scale':
        #     self.src_embed = MaxtrixScaleEmbedding(self.d_model)
        # elif self.prob_embed == 'nn.embed':
        #     self.src_embed = nn.Embedding(source_shape + 1, d_model, padding_idx = src_pad_token)
        #     assert(self.src_embed.padding_idx == 0)
        # else:
        #     raise ValueError("Invalid source embedding scheme given!")
        
        #Always use start and stop tokens for target, only padding is optional
        tgt_emb_shape = target_shape + 3 if tgt_pad_token is not None else target_shape + 2
        self.tgt_embed = nn.Embedding(tgt_emb_shape, d_model, padding_idx = tgt_pad_token)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.out = nn.Linear(d_model, tgt_emb_shape)

    def _get_tgt_mask(self, size):
        #Generate a mask for the target to preserve autoregressive property. Note that the mask is 
        #   Additive for the PyTorch transformer
        mask = torch.tril(torch.ones(size, size, dtype = self.dtype, device = self.device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    # def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
    #     tgt_mask = self._get_tgt_mask(tgt.size(1)).to(src.device)
    #     if self.prob_embed != 'nn.embed':
    #         src_key_pad_mask = None
    #         src = src.to(self.dtype)
    #     else:
    #         src_key_pad_mask = (src == self.src_pad_token).bool().to(self.device)

    #     tgt_key_pad_mask = (tgt == self.tgt_pad_token).bool().to(self.device)
    #     src = self.src_embed(src) * math.sqrt(self.d_model)
    #     tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
    #     if self.include_pos:
    #         src = self.pos_encoder(src)
    #         tgt = self.pos_encoder(tgt)
    #     transformer_out = self.transformer(src, tgt, 
    #                                        tgt_mask = tgt_mask, 
    #                                        tgt_key_padding_mask = tgt_key_pad_mask,
    #                                        #NOTE: src_key_padding_mask causes issues when dealing with no_grad() context, see 
    #                                        #    https://github.com/pytorch/pytorch/issues/106630
    #                                        src_key_padding_mask = src_key_pad_mask)
    #     out = self.out(transformer_out)
    #     return out
    
    #TODO: Think about this design some more
    # #Sketch of what the forward function could look like
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        tgt_mask = self._get_tgt_mask(tgt.size(1)).to(src.device)
        src_embedded, src_key_pad_mask = self.src_embed_fn(src, self.src_embed, self.src_pad_token, self.pos_encoder)
        tgt_embedded, tgt_key_pad_mask = self.tgt_embed_fn(tgt, self.tgt_embed, self.tgt_pad_token, self.pos_encoder)
        transformer_out = self.transformer(src_embedded, tgt_embedded,
                                           tgt_mask = tgt_mask,
                                           tgt_key_padding_mask = tgt_key_pad_mask,
                                           src_key_padding_mask = src_key_pad_mask)
        out = self.out(transformer_out)
        return out
    
    
    def get_loss(self,
                 x: Tuple[Tensor, Tuple], 
                 y: Tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        """
        Unpacks the input and obtains the loss value
        Args:
            x: Tuple of a tensor (input) and the set of smiles strings (smiles)
            y: A tuple of a the shifted target tensor and full target tensor
            loss_fn: The loss function to use for the model, with the signature
                tensor, tensor -> tensor
        """
        inp, smiles = x
        shifted_y, full_y = y
        if isinstance(self.src_embed, nn.Embedding):
            pred = self.forward(inp.long().to(self.device), 
                            shifted_y.to(self.device))
        else:
            pred = self.forward(inp.float().to(self.device), 
                            shifted_y.to(self.device))
        pred = pred.permute(0, 2, 1)
        loss = loss_fn(pred, full_y.to(self.device))
        return loss