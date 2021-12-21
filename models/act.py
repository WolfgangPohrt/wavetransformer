from torch.nn import Module
from torch import Tensor
from typing import ForwardRef, Optional, Union, Tuple, List

from torch import Tensor, zeros, cat as torch_cat
from torch.nn import Module, Linear, Softmax, Embedding, DataParallel
from torch.nn.functional import softmax

from modules import WaveBlock, WaveNetEncoder10
from modules.transformer import Transformer
from modules.transformer_block import TransformerBlock
from modules.positional_encoding import PositionalEncoding
import torch
import torch.nn.functional as F
from modules.decode_utils import greedy_decode, topk_sampling
from modules.beam import beam_decode, beam_decode_c
from modules.sampling import top_k_top_p_sampling
import gc
from models.ast_models import ASTModel

class AC_Transformer(Module):
    """
    AC_Transformer: Model with audioset/imagenet pretrained transformer encoder.
        :param input_tdim: input time dimention
        :type inpit_tdim: int
        :param num_layers_decoder: Number of transformer blocks
        :type num_layers_decoder: int
        :param num_heads_decoder: Number of attention heads in each MHA
        :type num_heads_decoder: int
        :param n_features_decoder: number of features for transformer
        :type n_features_decoder: int
        :param n_hidden_decoder: hidden dimension of transformer 
        :type n_hidden_decoder: int
        :param nb_classes: vocabulary size 
        :type nb_classes: int
        :param dropout_decoder: dropout rate in decoder
        :type dropout_decoder: float
        :param beam_size: beam size (<1: greedy, >1: beam search) 
        :type beam_size: int
    """


    def __init__(self,
                input_tdim: int,
                num_layers_decoder: int,
                num_heads_decoder: int,
                n_hidden_encoder: int,
                n_features_decoder: int,
                n_hidden_decoder: int,
                nb_classes: int,
                dropout_decoder: float,
                beam_size: int):
        super(AC_Transformer, self).__init__()
                
        self.max_length: int = 22
        self.nb_classes: int = nb_classes
        self.beam_size = beam_size

        self.encoder =  ASTModel(input_tdim=input_tdim, audioset_pretrain=True)
        self.sublayer_decoder: Module = TransformerBlock(
            n_features=n_features_decoder,
            n_hidden=n_hidden_decoder,
            num_heads=num_heads_decoder,
            nb_classes=self.nb_classes,
            dropout_p=dropout_decoder
        )

        self.decoder: Module = Transformer(
            layer=self.sublayer_decoder,
            num_layers=num_heads_decoder,
            nb_classes=self.nb_classes,
            n_features=n_features_decoder,
            dropout_p=dropout_decoder)

        self.embeddings: Embedding = Embedding(
            num_embeddings=self.nb_classes,
            embedding_dim=n_features_decoder)

        self.projection: Linear = Linear(n_hidden_encoder, n_features_decoder)

        self.classifier: Linear = Linear(
            in_features=n_features_decoder,
            out_features=self.nb_classes)

    def forward(self, x, y):
        y = y.permute(1, 0)[:-1]
        encoder_output: Tensor = self.encoder(x)
        encoder_output = self.projection(encoder_output)
        encoder_output = encoder_output.permute(1, 0, 2)
        word_embeddings: Tensor = self.embeddings(y)
        decoder_output: Tensor = self.decoder(
            word_embeddings,
            encoder_output,
            attention_mask=None
        )
        
        out: Tensor = self.classifier(decoder_output)
        # torch.cuda.empty_cache()
        # gc.collect()
        return out