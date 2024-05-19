import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

class UserGRU4Rec(torch.nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, emb_size, hidden_size, num_layers, dropout_prob):
        super(UserGRU4Rec, self).__init__()

        # load parameters info
        self.embedding_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob


        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq_emb, item_seq_len):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


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
        text_encoders_candidates = ['body', 'abstract']

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
        self.encoder = UserGRU4Rec(args.word_embedding_dim, int(args.word_embedding_dim/4), 1, dropout_prob=0.2)

    def forward(self, log_vec, log_mask, mask=None):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        item_seq_len = torch.sum(log_mask, dim=-1)
        vec = self.encoder(log_vec, item_seq_len)

        return vec


class GRU4Rec(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 args,
                 bert_model):
        super(GRU4Rec, self).__init__()
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

