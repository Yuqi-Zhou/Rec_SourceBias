import os
import utils
import torch
import logging
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from Dataset import NewsDataset, BehaviorsDataset, UserDataset
from Dataset import BaseDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class Trainer_main:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
        self.config = AutoConfig.from_pretrained(args.plm_model, output_hidden_states=True)
        args.word_embedding_dim = self.config.hidden_size
        bert_model = AutoModel.from_pretrained(args.plm_model, config=self.config)

        for name,param in bert_model.named_parameters():
            param.requires_grad = False

        model = utils.get_model(args.model_type)
        self.model = model(args, bert_model)

        state_dict = self.model.state_dict()
        self.save_state_dict = {}
        for key, param in state_dict.items():
            if 'bert' not in key:
                self.save_state_dict[key] = param

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.last_ratio = 0

    def save_model(self, path):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
            }, path)
        logging.info(f"Model saved to {path}")


    def train_epoch(self, dataloader):
        loss = 0.0
        accuary = 0.0  
        for cnt, batch  in enumerate(tqdm(dataloader)):
            candidate_news_id, clicked_news_id, targets, log_mask = batch["candidate_news_id"], batch["clicked_news_id"], batch["targets"].to(self.device), batch["log_mask"].to(self.device)
            candidate_news = torch.stack([torch.stack([self.text2vector[new].to(self.device) for new in bs]) for bs in batch["candidate_news_id"]]).to(self.device)
            candidate_news = candidate_news.view(-1, 1 + self.args.negative_sampling_ratio, self.args.word_embedding_dim)
            clicked_news_id = torch.stack([torch.stack([self.text2vector[new].to(self.device) for new in bs]) for bs in batch["clicked_news_id"]]).to(self.device)
            log_vec = clicked_news_id.view(-1, self.args.user_log_length,
                                self.args.word_embedding_dim)
            user_vector = self.model.user_encoder(log_vec, log_mask)
            y_hat = torch.bmm(candidate_news, user_vector.unsqueeze(-1)).squeeze(dim=-1)
            bz_loss = self.model.criterion(y_hat, targets)
            accuary += utils.acc(targets, y_hat)
            self.optimizer.zero_grad()
            bz_loss.backward()
            loss += bz_loss.cpu().detach().data.float()
            self.optimizer.step()
            
            if cnt % self.args.log_steps == 0:
                logging.info(
                    'Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                         cnt * self.args.batch_size, loss.data / cnt,
                        accuary / cnt))
        return loss / cnt

    def get_news2vector(self, news_file):
        news2vector = {}
        news_dataset = NewsDataset(self.args, news_file)
        news_dataloader = news_dataset.get_dataloader(batch_size=self.args.batch_size, \
            shuffle=False, num_workers=8)
        for batch in tqdm(news_dataloader, desc="Calculating vectors for news"):
            news_ids =  batch["id"] # batch, 1
            if any(id not in news2vector for id in news_ids):
                news_vector = self.model.get_news_vector(batch["news"].to(self.device))
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector.detach().to('cpu')
        news2vector['PADDED_NEWS'] = torch.zeros(self.args.word_embedding_dim)
        return news2vector

    def get_new_model(self):
        model = utils.get_model(self.args.model_type)
        model = model(self.args, None)
        model.load_state_dict(self.save_state_dict, strict=False)
        self.optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model
    
    def run(self):
        logging.info('Training...')

        self.model.train()
        self.model = self.model.to(self.device)
        self.human_text2vector = self.get_news2vector(self.args.human_news_file)

        self.text2vector = {}
        for k, v in self.human_text2vector.items():
            self.text2vector[k+'-human'] = v

        self.text2vector['PADDED_NEWS'] = self.human_text2vector['PADDED_NEWS']

        self.model = self.get_new_model()
        self.model.train()
        self.model = self.model.to(self.device)
        seed_torch(self.args.seed)
        for ep in range(self.args.epochs):
            dataset = BaseDataset(self.args, self.args.behaviors_file, None, None, None, epoch=0)
            dataloader = dataset.get_dataloader(batch_size=self.args.batch_size, shuffle=False, num_workers=8)

            loss = self.train_epoch(dataloader)
            logging.info(f"EPOCH[{ep + 1}] LOSS:{loss}")

            ckpt_path = os.path.join(self.args.ckpt_dir, f'epoch-{ep+1}.pt')
            self.save_model(ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
