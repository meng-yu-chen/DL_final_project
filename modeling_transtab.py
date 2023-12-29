import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

import constants

class TransTabWordEmbedding(nn.Module):

     # word string -> embedding (for col names and categorical features)
    def __init__(self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
    ) -> None:

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # initialize with normal distribution
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:

        # token id -> word embedding
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings =  self.dropout(embeddings)

        return embeddings

class TransTabNumEmbedding(nn.Module):

    # for numericl feature: (col_name -> word embedding) * numerical value
    def __init__(self, hidden_dim) -> None:
        
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # add bias
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        
        # initialize bias with uniform distribution, a - lower bound, b - upper bound
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        '''args:
            num_col_emb: numerical column embedding, (# numerical columns, emb_dim)
            x_num_ts: numerical features, (bs, emb_dim)
            num_mask: the mask for NaN numerical features, (bs, # numerical columns)
        '''

        #print(f'num_col_emb.shape: {num_col_emb.shape}')
        #print(f'x_num_ts.shape: {x_num_ts.shape}')
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

#Process input dataframe to input indices towards transtab encoder
class TransTabFeatureExtractor:
    
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
    ) -> None:
        
        '''args
            categorical_columns: a list of categories feature names
            numerical_columns: a list of numerical feature names
            binary_columns: a list of yes or no feature names, accept binary indicators like
                                (yes,no); (true,false); (0,1).
            disable_tokenizer_parallel: true if use extractor for collator function in torch.DataLoader
            ignore_duplicate_cols: check if exists one col belongs to both cat/num or cat/bin or num/bin,
                if set `true`, the duplicate cols will be deleted, else throws errors.
        '''

        # load pretrained tokenizer weight
        if os.path.exists('./transtab/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./transtab/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512
        
        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        # feature col names
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        # transfer to list type
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))


        # check if column exists overlap
        # return whether to overlap, overlapping columns
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)
        
    # feature -> embedding
    def __call__(self, x, shuffle=False) -> Dict: 

        '''parameters
            x: pd.DataFrame, with column names and features.
            shuffle: bool, if shuffle column order during the training.
        '''
        '''returns
            encoded_inputs: a dict with {
                    'x_num': tensor contains numerical features,
                    'num_col_input_ids': tensor contains numerical column tokenized ids,
                    'x_cat_input_ids': tensor contains categorical column + feature ids,
                    'x_bin_input_ids': tesnor contains binary column + feature ids,
                }
        '''

        encoded_inputs = {
            'x_num':None,
            'num_col_input_ids':None,
            'x_cat_input_ids':None,
            'x_bin_input_ids':None,
        }
        
        col_names = x.columns.tolist()

        # col name
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        #print(f"data_cfg['bin']: {bin_cols}")
        #print(f"data_cfg['cat']: {cat_cols}")
        #print(f"data_cfg['num']: {num_cols}")
        #print(f'x: {x}')

        #raise

        #for idx in range(len(num_cols)):
        #    print(f'num_cols[idx]: {num_cols[idx]}')
        #   print(f'x[v4]: {x["v4"]}')
        #    print(f'idx: {x[num_cols[idx]]}')

        #print(f'num_cols: {num_cols}')
        #raise

        # if none of col type is assigned, take all columns as categorical columns
        if len(cat_cols+num_cols+bin_cols) == 0:
            cat_cols = col_names

        # shuffle the col names
        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        
        # mask nan value in numerical columns
        if len(num_cols) > 0:

            # pick numerical col values
            x_num = x[num_cols]
            x_num.replace('?',np.nan)
            #print(f'x_num: {x_num}')

            # replace missing value and ? as 0
            x_num = x_num.fillna(0) 

            # transfer to tensor type
            x_num_ts = torch.tensor(np.array(x_num.values, dtype=float))
            
            # numerical col name-> token id
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            
            # numerical values
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            
            # mask out attention
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_cat.replace('?',np.nan)
            #print(f'x_cat: {x_cat}')
        
            # replace missing value and ? as ''
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')

            # categorical col name + ' ' + categorical col value
            x_cat = x_cat.apply(lambda x: x.name + ' '+ x) * x_mask # mask out nan features
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            # categorical col name & feature-> token id
            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        
        if len(bin_cols) > 0:
             #  x_bin should already be integral (binary values in 0 & 1)
            x_bin = x[bin_cols] 

            # categorical col name + ' ' 
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            
            # binary col name-> token id
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            if x_bin_ts['input_ids'].shape[1] > 0: # not all false
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']

        return encoded_inputs

    # save the feature extractor configuration to local dir
    def save(self, path):
        
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    # load pretrained feature extractor configuration from local dir
    def load(self, path):

        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        logger.info(f'load feature extractor from {coltype_path}')

    # update cat/num/bin column name
    def update(self, cat=None, num=None, bin=None):

        # update categorical col name
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        # update numerical col name
        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        # update biinary col name
        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        # check for column overlap
        # return whether to overlap, overlapping columns
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning('No cat/num/bin cols specified, will take ALL columns as categorical! Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

# TransTabFeatureExtractor -> TransTabFeatureProcessor
# process inputs from feature extractor to map them to embeddings
class TransTabFeatureProcessor(nn.Module):
    
    def __init__(self,
        vocab_size=None,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        device='cuda:0',
        ) -> None:
        '''args
            categorical_columns: a list of categories feature names
            numerical_columns: a list of numerical feature names
            binary_columns: a list of yes or no feature names, accept binary indicators like
                                (yes,no); (true,false); (0,1).
        '''

        super().__init__()

        # string token input_ids -> embedding
        self.word_embedding = TransTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id
        )

        # numerical feature -> embedding
        self.num_embedding = TransTabNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    # get average embedding
    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(1) / att_mask.sum(1,keepdim=True).to(embs.device)
            return embs

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        cat_att_mask=None,
        x_bin_input_ids=None,
        bin_att_mask=None,
        **kwargs,
        ) -> Tensor:
        '''args
            x: pd.DataFrame with column names and features.
            shuffle: if shuffle column order during the training.
            num_mask: indicate the NaN place of numerical features, 0: NaN 1: normal.
        '''

        # feature embedding
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        # handle numerical feature
        # (feature name -> embedding) * numerical value
        if x_num is not None and num_col_input_ids is not None:
            # numerical feature name -> embedding
            num_col_emb = self.word_embedding(num_col_input_ids.to(self.device)) # number of cat col, num of tokens, embdding size
            x_num = x_num.to(self.device)
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
            
            # (feature name embedding, numeric value)
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)
            
            # pass linear
            num_feat_embedding = self.align_layer(num_feat_embedding)

        # handle categorical feature
        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            # pass linear
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        # handle binary feature
        if x_bin_input_ids is not None:

            # all feature value are false, padding zero
            if x_bin_input_ids.shape[1] == 0: 
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0],dtype=int)[:,None]
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

        # concat all embeddings
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]
        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding]
            att_mask_list += [bin_att_mask]

        # no data -> error
        if len(emb_list) == 0: 
            raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        
        # hole row data embedding
        all_feat_embedding = torch.cat(emb_list, 1).float()
        
        # hole row data attention_mask
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}

# get activation function
def _get_activation_fn(activation):
    
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))

# gated transformer
class TransTabTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=2048, 
        dropout=0.1, 
        activation=F.relu,
        layer_norm_eps=1e-5, 
        batch_first=True, 
        norm_first=False,
        device='cuda:0', 
        dtype=None, 
        use_layer_norm=True
    ) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            batch_first=batch_first,
            **factory_kwargs
        )
        
        # Implementation of Feedforward model
        # (d_model, dim_feedforward)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        # (dim_feedforward, d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # set activation function
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:

        # pass sigmoid
        g = self.gate_act(self.gate_linear(x))

        # (d_model, dim_feedforward)
        h = self.linear1(x)

        # add gate
        h = h * g 

        # (dim_feedforward, d_model)
        h = self.linear2(self.dropout(self.activation(h)))

        return self.dropout2(h)

    # set state
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    # pass the input through the encoder layer
    def forward(self, src, src_mask= None, src_key_padding_mask= None, is_causal=None, **kwargs) -> Tensor:
        
        """Args
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

            Shape:
                see the docs in Transformer class.
        """
    
        x = src
        if self.use_layer_norm:

            # do layer normalization first -> multihead attention / feedforward
            if self.norm_first:
                # residual
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                # residual
                x = x + self._ff_block(self.norm2(x))
            else:
                # residual
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                # residual
                x = self.norm2(x + self._ff_block(x))

        else: 
            # not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)

        return x

# Build a feature encoder that maps inputs tabular samples to embeddings
class TransTabInputEncoder(nn.Module):
    '''parameters
        cat_cols: a list of categories feature names
        num_cols: a list of numerical feature names
        bin_cols: a list of yes or no feature names, accept binary indicators like (yes,no); (true,false); (0,1).
        ignore_duplicate_cols: check if exists one col belongs to both cat/num or cat/bin or num/bin,
            if set `true`, the duplicate cols will be deleted, else throws errors.

        hidden_dim: int, the dimension of hidden embeddings.
        hidden_dropout_prob: float, the dropout ratio in the transformer encoder.
    
        device: str, the device, ``"cpu"`` or ``"cuda:0"``.
    '''

    def __init__(self,
        feature_extractor,
        feature_processor,
        device='cuda:0',
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    # encode tabular data -> embedding
    def forward(self, x):
        '''Parameters
            x: pd.DataFrame, with column names and features.  
        '''

        # 分類各個f eature type,然後轉成 id
        tokenized = self.feature_extractor(x)

        # 所有 col feature 轉成 embedding 後,全部 concat 起來
        embeds = self.feature_processor(**tokenized)

        return embeds
    
    def load(self, ckpt_dir):
        # load feature extractor weight
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

# final embedding after pass gated transformer
class TransTabEncoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
    ):
        
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
            TransTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            ]
            )
        if num_layer > 1:
            encoder_layer = TransTabTransformerLayer(d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,
            )

            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer-1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        
        # embedding: bs, num_token, hidden_dim
        outputs = embedding

        # pass through stack gated transformer
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        
        return outputs

class TransTabLinearClassifier(nn.Module):
    def __init__(
        self,
        num_class,
        hidden_dim=128
    ) -> None:
        
        super().__init__()
        # less than two classes
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        # pick the cls token to predict
        x = x[:,0,:] 
        x = self.norm(x)
        logits = self.fc(x)
        return logits

# projection dimension
class TransTabProjectionHead(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        projection_dim=128
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h

# add a learnable cls token embedding at the end of each sequence
class TransTabCLSToken(nn.Module):

    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))

        # initialize cls embedding with uniform distribution
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    # add cls embedding to seq embedding
    def forward(
        self, 
        embedding, 
        attention_mask=None, 
        **kwargs
    ) -> Tensor:
        
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}

        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        
        return outputs

class TransTabModel(nn.Module):
    '''
        The base transtab model for downstream tasks like contrastive learning, binary classification, etc.
        All models subclass this basemodel and usually rewrite the ``forward`` function. Refer to the source code of
        :class:`transtab.modeling_transtab.TransTabClassifier` or :class:`transtab.modeling_transtab.TransTabForCL` for the implementation details.
    '''

    '''Parameters
        cat_cols: list, a list of categories feature names
        num_cols: list, a list of numerical feature names
        bin_cols: list, a list of yes or no feature names, accept binary indicators like (yes,no); (true,false); (0,1).

        hidden_dim: int, the dimension of hidden embeddings
        num_layer: int, the number of transformer layers used in the encoder
        num_attn_head: int, the numebr of heads of multihead self-attention layer in the transformers
        ffn_dim: int, the dimension of feed-forward layer in the transformer layer
        hidden_dropout_prob: float, the dropout ratio in the transformer encoder

        activation: str, the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
        device: str, the device, ``"cpu"`` or ``"cuda:0"``
    '''

    '''returns
        A TransTabModel model
    '''

    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
    ) -> None:

        super().__init__()
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns

        # transfer to list type
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        # extractor different type of col and do preprocessing (word embedding, handle missing value...)
        if feature_extractor is None:
            feature_extractor = TransTabFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        # generate processed col embedding
        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
        )
        
        # feature extractor -> feature_processor, get input embedding for TransTabEncoder 
        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        # final embedding after pass gated transformer
        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

        # add CLS token
        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        '''Parameters
            x: pd.DataFrame, a batch of samples stored in pd.DataFrame.
            y: pd.Series, the corresponding labels for each sample in ``x``. ignored for the basemodel
        '''
        '''return
            final_cls_embedding: torch.Tensor, the [CLS] embedding at the end of transformer encoder
        '''
        # process col
        embeded = self.input_encoder(x)

        # add cls token
        embeded = self.cls_token(**embeded)

        # pass gated transformer
        encoder_output = self.encoder(**embeded)

        # pick cls token
        final_cls_embedding = encoder_output[:,0,:]
        return final_cls_embedding

    # load the model state_dict and feature_extractor configuration from the ``ckpt_dir``
    def load(self, ckpt_dir):
        '''Parameters
            ckpt_dir: str, the directory path to load.
        '''
        '''return 
            None
        '''

        # load model weight state dict
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    # Save the model state_dict and feature_extractor configuration to the ``ckpt_dir``
    def save(self, ckpt_dir):
        '''Parameters
            ckpt_dir: str, the directory path to save.
        '''
        '''return 
            None
        '''
       
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    # update model parameter
    def update(self, config):

        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))

        # cols name appears more than 1 time
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        
        # whether to overlap, overlapping columns
        return org_length == unq_length, duplicate_cols

    # handle duplicate_column
    def _solve_duplicate_cols(self, duplicate_cols):

        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

    # adjust class number
    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class

            # adjust FC dimension
            self.clf = TransTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)

            # define loss function
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')

# transtab model for downstream classification tasks
class TransTabClassifier(TransTabModel):
    '''
        The classifier model subclass from :class:`transtab.modeling_transtab.TransTabModel`.
    '''

    '''Parameters
        cat_cols: list, a list of categories feature names
        num_cols: list, a list of numerical feature names
        bin_cols: list, a list of yes or no feature names, accept binary indicators like (yes,no); (true,false); (0,1).

        feature_extractor: TransTabFeatureExtractor, a feature extractor to tokenize the input tables. if not passed the model will build itself.
        num_class: int, number of output classes to be predicted.
        
        hidden_dim: int, the dimension of hidden embeddings
        num_layer: int, the number of transformer layers used in the encoder
        num_attn_head: int, the numebr of heads of multihead self-attention layer in the transformers
        ffn_dim: int, the dimension of feed-forward layer in the transformer layer
        hidden_dropout_prob: float, the dropout ratio in the transformer encoder

        activation: str, the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
        device: str, the device, ``"cpu"`` or ``"cuda:0"``
    '''

    '''returns
        A TransTabClassifier model
    '''

    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
    ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        
        #define loss function
        if self.num_class > 2:
             # reduction = none, 表示直接傳回n分樣本的loss
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

        
    # Make forward pass given the input feature ``x`` and label ``y`` (optional
    def forward(self, x, y=None):
        '''Parameters
            x: pd.DataFrame or dict,
               pd.DataFrame: a batch of raw tabular samples; 
               dict: the output of TransTabFeatureExtractor.
            y: pd.Series
               the corresponding labels for each sample in ``x``. 
               if label is given, the model will return the classification loss by ``self.loss_fn``.
        '''

        '''return
            logits: torch.Tensor, the [CLS] embedding at the end of transformer encoder.

            loss: torch.Tensor or None
            the classification loss.
        '''
        
        #print(f'x: {x}')
        #raise

        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x)
        else:
            raise ValueError(f'TransTabClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')
        
        # generate processed col embedding
        outputs = self.input_encoder.feature_processor(**inputs)
        
        # add cls token
        outputs = self.cls_token(**outputs)

        # go through transformers, get the first cls embedding
        encoder_output = self.encoder(**outputs) # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf(encoder_output)

        # if label is given, compute classification loss
        if y is not None:
            if self.num_class == 2:
                y_ts = torch.tensor(np.array(y.values, dtype=float)).to(self.device)
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(self.device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss

# transtab model for downstream contrasstive learning
class TransTabForCL(TransTabModel):
    '''
        The contrasstive learning model subclass from :class:`transtab.modeling_transtab.TransTabModel`.
    '''

    '''Parameters
        cat_cols: list, a list of categories feature names
        num_cols: list, a list of numerical feature names
        bin_cols: list, a list of yes or no feature names, accept binary indicators like (yes,no); (true,false); (0,1).

        feature_extractor: TransTabFeatureExtractor, a feature extractor to tokenize the input tables. if not passed the model will build itself.
        proj_dim: int, the dimension of projection head on the top of encoder.

        hidden_dim: int, the dimension of hidden embeddings
        num_layer: int, the number of transformer layers used in the encoder
        num_attn_head: int, the numebr of heads of multihead self-attention layer in the transformers
        ffn_dim: int, the dimension of feed-forward layer in the transformer layer
        hidden_dropout_prob: float, the dropout ratio in the transformer encoder

        overlap_ratio: float, the overlap ratio of columns of different partitions when doing subsetting
        num_partition: int, the number of partitions made for vertical-partition contrastive learning
        supervised: bool, whether or not to take supervised VPCL, otherwise take self-supervised VPCL

        temperature: float, temperature used to compute logits for contrastive learning
        base_temperature: float, base temperature used to normalize the temperature

        activation: str, the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
        device: str, the device, ``"cpu"`` or ``"cuda:0"``
    '''

    '''returns
        A TransTabForCL model
    '''

    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.1,
        num_partition=2,
        supervised=True,
        temperature=10,
        base_temperature=10,
        activation='relu',
        device='cuda:0',
        **kwargs,
    ) -> None:
        
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        '''Parameters
            x: pd.DataFrame or dict,
               pd.DataFrame: a batch of raw tabular samples; 
               dict: the output of TransTabFeatureExtractor.

            y: pd.Series,
               the corresponding labels for each sample in ``x``. 
               if label is given, the model will return                the classification loss by ``self.loss_fn``.
        '''

        '''return
            logits: None, this CL model does NOT return logits.

            loss: torch.Tensor, the supervised or self-supervised VPCL loss.
        '''

        # do positive sampling
        feat_x_list = []
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_positive_pairs(x, self.num_partition)
            for sub_x in sub_x_list:
                # encode two subset feature samples
                feat_x = self.input_encoder(sub_x)

                # add cls token
                feat_x = self.cls_token(**feat_x)

                # go through gated transformers, get the first cls embedding
                feat_x = self.encoder(**feat_x)

                # take cls embedding
                feat_x_proj = feat_x[:,0,:] 

                # linear projection
                feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim
                feat_x_list.append(feat_x_proj)

        elif isinstance(x, dict):
            # pretokenized inputs
            for input_x in x['input_sub_x']:
                feat_x = self.input_encoder.feature_processor(**input_x)
                
                # add cls token 
                feat_x = self.cls_token(**feat_x)

                # go through gated transformers, get the first cls embedding
                feat_x = self.encoder(**feat_x)

                # pick cls embedding
                feat_x_proj = feat_x[:, 0, :]

                # linear projection
                feat_x_proj = self.projection_head(feat_x_proj)
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f'expect input x to be pd.DataFrame or dict(pretokenized), get {type(x)} instead')

        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, n_view, emb_dim

        if y is not None and self.supervised:
            # take supervised loss
            y = torch.tensor(y.values, device=feat_x_multiview.device)
            loss = self.supervised_contrastive_loss(feat_x_multiview, y)
        else:
            # compute cl loss (multi-view InfoNCE loss)
            loss = self.self_supervised_contrastive_loss(feat_x_multiview)
        return None, loss

    def _build_positive_pairs(self, x, n):
        x_cols = x.columns.tolist()

        # 切成 n 分
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []

        # overlap each split of cloumn
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
            elif overlap >0 and i == n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)

        return sub_x_list

    # compute cosine similarity
    def cos_sim(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        
        # 矩陣相乘
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    # Compute the self-supervised VPCL loss (without laabel)
    def self_supervised_contrastive_loss(self, features):
        '''Parameters
            features: torch.Tensor, the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.
        '''

        '''return
            loss: torch.Tensor, the computed self-supervised VPCL loss.
        '''

        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        # [[0,1],[2,3]] -> [0,2,1,3]
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out contrast cases
        # scatter 函數的一個典型應用就是在分類問題中，將目標標籤轉換為one-hot編碼形式
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    # Compute the supervised VPCL loss
    def supervised_contrastive_loss(self, features, labels):
        '''Parmeters
            features: torch.Tensor, the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.
            labels: torch.Tensor, the class labels to be used for building positive/negative pairs in VPCL.
        '''

        '''return
            loss: torch.Tensor, the computed VPCL loss.
        '''

        # contiguous 為返回具有連續記憶體的相同資料型態的 tensor, 因為 view 操作要求 tensor 的內存連續存儲
        # 調用 contiguous，然後方可使用 view 對維度進行變形
        labels = labels.contiguous().view(-1,1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        # n partition
        contrast_count = features.shape[1]

        # unbind 此方法就是將我們的 input 從 dim 進行切片，並傳回切片的結果，傳回的結果裡面沒有dim這個維度
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)

        # contrast_mode == 'all'
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
       
        # mask-out self-contrast cases
        # scatter函數的一個典型應用就是在分類問題中，將目標標籤轉換為one-hot編碼形式
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
