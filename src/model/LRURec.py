import torch
from torch import nn
import torch.nn.functional as F
import os
import math
import numpy as np
HIDDEN_SIZE=768

class LRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LRUModel()
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, x, masks):
        return self.model(x, masks)

class LRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = 1

        self.lru_blocks = nn.ModuleList([LRUBlock() for _ in range(layers)])

    def forward(self, x, mask):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))
        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        # x = x[:, -1, :]  # B x L x D (64)
        item_seq_len = torch.sum(mask, dim=-1)
        x = self.gather_indexes(x, item_seq_len - 1 + 2 ** log2_L - mask.size(1))
        return x

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
  

class LRUBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = HIDDEN_SIZE
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=0.2)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size*4)
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x
    

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)




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
        self.hidden_layer = args.hidden_layer
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
        self.encoder = LRU()

    def forward(self, log_vec, log_mask, mask=None):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        vec = self.encoder(log_vec, log_mask)

        return vec


class LRURec(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 args,
                 bert_model):
        super(LRURec, self).__init__()
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

