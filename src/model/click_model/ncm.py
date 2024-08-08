import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.layers import TransformerEncoder

class Transforemer_NeuralClickModel(nn.Module):
    def __init__(self, args):
        super().__init__()
                
        self.emb_dim = args.word_embedding_dim
        self.n_layers = 1
        self.n_heads = 2
        self.hidden_size = self.emb_dim
        self.inner_size = 128
        self.hidden_dropout_prob = 0.2
        self.attn_dropout_prob = 0.2
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02
        self.max_seq_length = 2*(args.negative_sampling_ratio + 1)
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        ) 
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
        self.out_layer = nn.Linear(2*self.emb_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, item_seq, user_embs):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(torch.ones(item_seq.shape[0], item_seq.shape[1]).cuda())
        position_embedding = self.position_embedding(position_ids)
        item_emb = item_seq
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        mask = torch.ones((item_seq.shape[0], item_seq.shape[1])).cuda()
        extended_attention_mask = self.get_attention_mask(mask, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        
        user_embs = user_embs.unsqueeze(1).repeat(1, item_seq.shape[1], 1)
        features = torch.cat([output, user_embs], dim=-1)
        y = self.out_layer(features)
        return y.squeeze() 
    
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


class NeuralClickModel(nn.Module):
    def __init__(self, args):
        super().__init__()
                
        self.emb_dim = args.word_embedding_dim
        self.rnn_layer = nn.GRU(
            input_size=self.emb_dim, 
            hidden_size=self.emb_dim, 
            batch_first=True
        )
        self.out_layer = nn.Linear(self.emb_dim, 1)


    def forward(self, items, user_embs):
        rnn_out, _ = self.rnn_layer(items, user_embs.unsqueeze(0))
        y = self.out_layer(rnn_out)
        return y.squeeze()

class MF(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, items, user_embs):
        user_expanded = user_embs.unsqueeze(1).expand_as(items)
        dot_product = torch.sum(items * user_expanded, dim=-1)
        return dot_product

class LogisticRegression(nn.Module):
    def __init__(self, args):
        super().__init__()
                
        self.emb_dim = args.word_embedding_dim
        self.out_layer = nn.Linear(2*self.emb_dim, 1)

    def forward(self, items, user_embs):
        _, itemn, _ = items.shape
        user_embs = user_embs.unsqueeze(1).repeat(1, itemn, 1)
        features = torch.cat([items, user_embs], dim=-1)
        y = self.out_layer(features)
        return y.squeeze()

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model

    def forward(self, text, mask=None):
        ids_length = text.size(1)
        text = text.view(-1, ids_length)
        batch_size, num_words = text.shape
        num_words = num_words // 3
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_type = torch.narrow(text, 1, num_words, num_words)
        text_attmask = torch.narrow(text, 1, num_words*2, num_words)
        hidden_states = self.bert_model(text_ids, token_type_ids=text_type, attention_mask=text_attmask)['hidden_states']
        word_emb = hidden_states[-1][:, 0, :]
        return word_emb
