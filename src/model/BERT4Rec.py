import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class UserBERT4Rec(torch.nn.Module):
    def __init__(self, max_seq_length):
        super(UserBERT4Rec, self).__init__()

        # load parameters info
        self.n_layers = 1
        self.n_heads = 2
        self.hidden_size = 768  # same as embedding_size
        self.inner_size = 128
        self.hidden_dropout_prob = 0.2
        self.attn_dropout_prob = 0.2
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02
        self.max_seq_length = max_seq_length

        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
        
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        # self.output_bias = nn.Parameter(torch.zeros(self.n_items))
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(torch.ones(item_seq.shape[0], item_seq.shape[1]).cuda())
        position_embedding = self.position_embedding(position_ids)
        # item_emb = self.item_embedding(item_seq)
        item_emb = item_seq
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        mask = torch.zeros((item_seq.shape[0], item_seq.shape[1])).cuda()
        for i, value in enumerate(item_seq_len):
            mask[i, :value] = 1
        extended_attention_mask = self.get_attention_mask(mask, bidirectional=True)
        # extended_attention_mask = item_mask
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output) # [B L H] 我们需要的是[B H]，所以要gather index
        output = self.gather_indexes(output, item_seq_len-1)
        return output 

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model

    def forward(self, text, mask=None):
        """
        Args:
            text: Tensor(batch_size) * num_words_text * embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_text
        batch_size, num_words = text.shape
        num_words = num_words // 3
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_type = torch.narrow(text, 1, num_words, num_words)
        text_attmask = torch.narrow(text, 1, num_words*2, num_words)
        hidden_states = self.bert_model(text_ids, token_type_ids=text_type, attention_mask=text_attmask)['hidden_states']
        word_emb = torch.mean(hidden_states[-1], dim=1)
        return word_emb

class NewsEncoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'abstract': args.num_words_abstract * 3,
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)])
            for key in self.attributes2length.keys()
        }
        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['abstract']

        self.text_encoders = nn.ModuleDict({
            k :
            TextEncoder(bert_model,)
        for k in args.news_attributes})

        self.newsname=[name for name in set(args.news_attributes) & set(text_encoders_candidates)]

    def forward(self, news):
        """
        Args:
        Returns:
            (shape) batch_size, news_dim
        """
        text_vectors = [
            self.text_encoders[name](
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]))
            for name in self.newsname
        ]

        all_vectors = text_vectors
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:

            final_news_vector = torch.mean(
                torch.stack(all_vectors, dim=1),
                dim=1
             )
        return final_news_vector


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.encoder = UserBERT4Rec(args.user_log_length)

    def forward(self, log_vec, log_mask, mask=None):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        item_seq_len = torch.sum(log_mask, dim=-1)
        vec = self.encoder(log_vec, item_seq_len)

        return vec


class BERT4Rec(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 args,
                 bert_model):
        super(BERT4Rec, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args,
                                        bert_model)

        self.user_encoder = UserEncoder(args)

        self.criterion = nn.CrossEntropyLoss()

    def get_prediction(self, candidate_news_vector, user_vector):
        score = torch.bmm(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(-1)).squeeze(dim=-1)
        return score

    def get_news_vector(self, input_ids):
        ids_length = input_ids.size(1)
        input_ids = input_ids.view(-1, ids_length)
        news_vec = self.news_encoder(input_ids)
        news_vec = news_vec.view(-1, self.args.word_embedding_dim)
        return news_vec

    def get_user_vector(self, log_ids, log_mask):
        log_vec = log_ids.view(-1, self.args.user_log_length,
                               self.args.word_embedding_dim)
        user_vector = self.user_encoder(log_vec, log_mask)
        return user_vector

    def forward(self,
                input_ids,
                log_ids,
                log_mask,
                targets=None,
                compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        ids_length = input_ids.size(2)
        input_ids = input_ids.view(-1, ids_length)
        news_vec = self.news_encoder(input_ids)
        news_vec = news_vec.view(-1, 1 + self.args.negative_sampling_ratio, self.args.word_embedding_dim)

        # batch_size, news_dim
        log_ids = log_ids.view(-1, ids_length)
        log_vec = self.news_encoder(log_ids)
        log_vec = log_vec.view(-1, self.args.user_log_length,
                               self.args.word_embedding_dim)

        user_vector = self.user_encoder(log_vec, log_mask)

        # batch_size, 2
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score

