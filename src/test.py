import os
import utils
import json
import torch
import logging
import numpy as np
import torch.optim as optim
# import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from Dataset import NewsDataset, BehaviorsDataset, UserDataset

class Tester:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
        self.config = AutoConfig.from_pretrained(args.plm_model, output_hidden_states=True)
        args.word_embedding_dim = self.config.hidden_size
        self.args = args
        bert_model = AutoModel.from_pretrained(args.plm_model, config=self.config)
        model = utils.get_model(args.model_type)
        self.model = model(args, bert_model)

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

    def get_user2vector(self, human_newsvector, llm_newsvector, ratio):
        user2vector = {}
        user_dataset = UserDataset(self.args, self.args.test_behaviors_file, human_newsvector, llm_newsvector, ratio)
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

    def mix_news(self, human_news2vector, llm_news2vector, ratio):
        blended_dict = {}
        for key in human_news2vector.keys():
            blended_value = human_news2vector[key] if np.random.rand() > ratio else llm_news2vector[key]
            blended_dict[key] = blended_value
        return blended_dict

    def evaluates(self):
        checkpoint = torch.load(self.args.load_ckpt_name)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        human_news2vector = self.get_news2vector(self.args.human_news_file)
        llm_news2vector = self.get_news2vector(self.args.llm_news_file)
        
        for ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            human_metrics, llm_metrics = utils.metrics('Human', ratio=ratio), utils.metrics('LLM', ratio=ratio)
            human_target_metrics, llm_target_metrics = utils.metrics('Human Target', ratio=ratio), utils.metrics('LLM Target', ratio=ratio)

            self.user2vector = self.get_user2vector(human_news2vector, llm_news2vector, ratio)
            behaviors_dataset = BehaviorsDataset(self.args, self.args.test_behaviors_file)
            behaviors_dataloader = behaviors_dataset.get_dataloader(batch_size=1, \
                shuffle=False, num_workers=8)
            for batch in tqdm(behaviors_dataloader,
                                desc="Calculating probabilities"):
                candidate_news = batch["candidate_news"]
                labels = np.array(batch["y_true"][0])
                if np.all(labels == 0) or np.all(labels == 1):
                    continue
                human_candidate_news_vector = torch.stack([human_news2vector[new] for new in candidate_news[0]]).numpy()
                llm_candidate_news_vector = torch.stack([llm_news2vector[new] for new in candidate_news[0]]).numpy()
                user_vector = [self.user2vector[clicked_news_string] for clicked_news_string in batch["clicked_news_string"]][0].numpy()
                human_y_pred = np.dot(human_candidate_news_vector, user_vector)
                llm_y_pred = np.dot(llm_candidate_news_vector, user_vector)
                if human_y_pred[0] == llm_y_pred[0]:
                    continue

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


    def run(self):
        self.evaluates()