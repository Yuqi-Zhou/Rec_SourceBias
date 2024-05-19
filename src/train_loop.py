import os
import utils
import torch
import logging
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from Dataset import NewsDataset, BehaviorsDataset, UserDataset, BaseDataset
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

class Trainer:
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

    def get_new_model(self):
        model = utils.get_model(self.args.model_type)
        model = model(self.args, None)
        model.load_state_dict(self.save_state_dict, strict=False)
        self.optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model

    def train_epoch(self, dataloader):
        loss = 0.0
        accuary = 0.0
        for cnt, batch in enumerate(tqdm(dataloader)):
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


    def train_debias_double_align_epoch(self, dataloader):
            loss = 0.0
            accuary = 0.0
            for cnt, batch  in enumerate(tqdm(dataloader)):
                candidate_news_id, clicked_news_id, targets, log_mask = batch["candidate_news_id"], batch["clicked_news_id"], batch["targets"].to(self.device), batch["log_mask"].to(self.device)
                candidate_news = torch.stack([torch.stack([self.text2vector[new] for new in bs]) for bs in batch["candidate_news_id"]]).to(self.device)
                random_candidate_news = torch.stack([torch.stack([self.rewrite_text2vector[new] for new in bs]) for bs in batch["candidate_news_id"]]).to(self.device)
                candidate_news = candidate_news.view(-1, 1 + self.args.negative_sampling_ratio, self.args.word_embedding_dim)
                random_candidate_news = random_candidate_news.view(-1, 1 + self.args.negative_sampling_ratio, self.args.word_embedding_dim)
                
                batch_size, seq_length, embedding_dim = candidate_news.shape
                
                clicked_news_id = torch.stack([torch.stack([self.text2vector[new] for new in bs]) for bs in batch["clicked_news_id"]]).to(self.device)
                random_clicked_news_id = torch.stack([torch.stack([self.rewrite_text2vector[new] for new in bs]) for bs in batch["clicked_news_id"]]).to(self.device)

                log_vec = clicked_news_id.view(-1, self.args.user_log_length, self.args.word_embedding_dim)
                random_log_vec = random_clicked_news_id.view(-1, self.args.user_log_length, self.args.word_embedding_dim)
                batch_size, seq_length, embedding_dim = log_vec.shape
                random_choice = torch.bernoulli(torch.full((batch_size, seq_length), self.args.disturb_ratio)).unsqueeze(-1).expand(-1, -1, embedding_dim)
                random_log_vec[random_choice == 0] = log_vec[random_choice == 0]
                
                user_vector = self.model.user_encoder(log_vec, log_mask)
                random_user_vector = self.model.user_encoder(random_log_vec, log_mask)
                
                y_hat = torch.bmm(candidate_news, user_vector.unsqueeze(-1)).squeeze(dim=-1)
                disturb_news_y_hat = torch.bmm(random_candidate_news, user_vector.unsqueeze(-1)).squeeze(dim=-1)
                distrub_users_y_hat = torch.bmm(candidate_news, random_user_vector.unsqueeze(-1)).squeeze(dim=-1)

                bz_loss = self.model.criterion(y_hat, targets)
                debias_news_loss = F.l1_loss(y_hat, disturb_news_y_hat)
                debias_users_loss = F.l1_loss(y_hat, distrub_users_y_hat)

                user_vector = user_vector.reshape(-1, self.args.word_embedding_dim)
                p = F.softmax(user_vector, dim=-1)
                p_log_p = - p * torch.log(p + 1e-10)
                entropy = torch.sum(p_log_p, dim=-1)
                reg_loss = torch.mean(entropy)

                random_user_vector = random_user_vector.reshape(-1, self.args.word_embedding_dim)
                p = F.softmax(random_user_vector, dim=-1)
                p_log_p = - p * torch.log(p + 1e-10)
                entropy = torch.sum(p_log_p, dim=-1)
                random_reg_loss = torch.mean(entropy)
                
                bz_loss = bz_loss + self.args.news_loss*debias_news_loss + self.args.user_loss*(debias_users_loss + reg_loss + random_reg_loss)

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


    def evaluates(self, test_epoch, ratio):
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        logging.info(f"Epoch: {test_epoch}")
        human_cnt, llm_cnt = 0, 0
        human_metrics, llm_metrics = utils.metrics('Human', ratio=ratio), utils.metrics('LLM', ratio=ratio)
        human_target_metrics, llm_target_metrics = utils.metrics('Human Target', ratio=ratio), utils.metrics('LLM Target', ratio=ratio)

        self.user2vector = self.get_user2vector(self.human_text2vector, self.llm_text2vector, self.args.test_behaviors_file, ratio)
        behaviors_dataset = BehaviorsDataset(self.args, self.args.test_behaviors_file)
        behaviors_dataloader = behaviors_dataset.get_dataloader(batch_size=1, \
            shuffle=False, num_workers=8)

        for batch in tqdm(behaviors_dataloader,
                            desc="Calculating probabilities"):
            candidate_news = batch["candidate_news"]
            labels = np.array(batch["y_true"][0])
            if np.all(labels == 0) or np.all(labels == 1):
                continue
            human_candidate_news_vector = torch.stack([self.human_text2vector[new] for new in candidate_news[0]]).numpy()
            llm_candidate_news_vector = torch.stack([self.llm_text2vector[new] for new in candidate_news[0]]).numpy()
            user_vector = [self.user2vector[clicked_news_string] for clicked_news_string in batch["clicked_news_string"]][0].numpy()
            human_y_pred = np.dot(human_candidate_news_vector, user_vector)
            llm_y_pred = np.dot(llm_candidate_news_vector, user_vector)

            if human_y_pred[0] == llm_y_pred[0]:
                continue
            
            human_cnt += np.sum((human_y_pred - llm_y_pred)[labels == 1] > 0)
            llm_cnt += np.sum((human_y_pred - llm_y_pred)[labels == 1] < 0)
            
            if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:
                y_pred_mix = np.concatenate((llm_y_pred, human_y_pred))
                label_human_mix = np.concatenate((np.zeros_like(labels), labels))
                label_lm_mix = np.concatenate((labels, np.zeros_like(labels)))
            else:
                y_pred_mix = np.concatenate((human_y_pred, llm_y_pred))
                label_human_mix = np.concatenate((labels, np.zeros_like(labels)))
                label_lm_mix = np.concatenate((np.zeros_like(labels), labels))

            human_metrics.update(labels, human_y_pred)
            llm_metrics.update(labels, llm_y_pred)
            human_target_metrics.update(label_human_mix, y_pred_mix)
            llm_target_metrics.update(label_lm_mix, y_pred_mix)

        human_metrics.print_result()
        llm_metrics.print_result()
        human_target_metrics.print_result()
        llm_target_metrics.print_result()
        logging.info(f"llm ratio: {llm_cnt/(human_cnt + llm_cnt)}")


    def get_user2vector(self, human_newsvector, llm_newsvector, behaviors_file, ratio):
        user2vector = {}
        user_dataset = UserDataset(self.args, behaviors_file, human_newsvector, llm_newsvector, ratio)
        user_dataloader = user_dataset.get_dataloader(batch_size=self.args.batch_size, \
            shuffle=False, num_workers=8)
        for batch in tqdm(user_dataloader,
                            desc="Calculating vectors for users"):
            user_strings = batch["clicked_news_string"]

            if any(user_string not in user2vector for user_string in user_strings):
                log_ids, log_mask = batch["log_ids"].to(self.device), batch["log_mask"].to(self.device)
                user_vector = self.model.get_user_vector(log_ids, log_mask)
                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector.detach().to('cpu')
        return user2vector

    def run(self):
        logging.info('Training...')
        self.model.train()
        self.model = self.model.to(self.device)

        self.human_text2vector = self.get_news2vector(self.args.human_news_file)
        self.llm_text2vector = self.get_news2vector(self.args.llm_news_file)

        self.text2vector = {}
        for k, v in self.human_text2vector.items():
            self.text2vector[k+'-human'] = v
        for k, v in self.llm_text2vector.items():
            self.text2vector[k+'-llm'] = v
        self.text2vector['PADDED_NEWS'] = self.human_text2vector['PADDED_NEWS']

        if self.args.debias is True:
            self.llm_rewrite_text2vector = self.get_news2vector(self.args.llm_rewirte_news_file)
            self.rewrite_text2vector = {}
            for k, v in self.llm_text2vector.items():
                self.rewrite_text2vector[k+'-human'] = v
            for k, v in self.llm_rewrite_text2vector.items():
                self.rewrite_text2vector[k+'-llm'] = v
            self.rewrite_text2vector['PADDED_NEWS'] = self.llm_text2vector['PADDED_NEWS']
            
        for lep in range(self.args.loop_epochs):
            seed_torch(self.args.seed)
            self.model.eval()
            if lep % 2 == 0:
                user2vector = self.get_user2vector(self.human_text2vector, self.llm_text2vector, self.args.behaviors_file1, ratio=self.last_ratio)
            else:
                user2vector = self.get_user2vector(self.human_text2vector, self.llm_text2vector, self.args.behaviors_file2, ratio=self.last_ratio)
            self.model = self.get_new_model()
            self.model.train()
            self.model = self.model.to(self.device)

            for ep in range(self.args.epochs):
                self.model.train()
                if lep % 2 == 0:
                    dataset = BaseDataset(self.args, self.args.behaviors_file1, self.human_text2vector, self.llm_text2vector, user2vector, epoch=lep)
                    dataloader = dataset.get_dataloader(batch_size=self.args.batch_size, shuffle=False, num_workers=8)
                else:
                    dataset = BaseDataset(self.args, self.args.behaviors_file2, self.human_text2vector, self.llm_text2vector, user2vector, epoch=lep)
                    dataloader = dataset.get_dataloader(batch_size=self.args.batch_size, shuffle=False, num_workers=8)
                if self.args.debias is True:
                    loss = self.train_debias_double_align_epoch(dataloader)
                else:
                    loss = self.train_epoch(dataloader)
                logging.info(f"EPOCH[{ep + 1}] LOSS:{loss}")

            self.last_ratio = dataset.new_ratio
            self.evaluates(lep+1, self.last_ratio)
