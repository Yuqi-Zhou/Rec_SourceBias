import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils import data
import pandas as pd
from ast import literal_eval
import numpy as np
import swifter
import torch

use_cols = ['abstract_input_ids', 'abstract_token_type_ids', 'abstract_attention_mask']


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)
    
def get_click_index(score, label, eta=-1):
    indices = np.argsort(score)[::-1]
    sorted_label = label[indices]
    if eta != -1:
        rank_score = np.array([pow(1/(i+1), eta) for i in range(len(sorted_label))])
        rank = np.array([i for i in range(len(sorted_label))])
        rank_score = rank_score[sorted_label == 1]
        rank = rank[sorted_label == 1]
        normalize_score = rank_score / (np.sum(rank_score))
        selected_index = np.random.choice(range(len(rank)), p=normalize_score)
        select_index = indices[rank[selected_index]]
    else:
        rank = np.array([i for i in range(len(sorted_label))])
        rank = rank[sorted_label == 1]
        selected_index = 0
        select_index = indices[rank[selected_index]]
    return select_index

class BaseDataset(Dataset):
    def __init__(self, args, behaviors_path, human_news2vector, llm_news2vector, user2vector, epoch, ratio=None):
        super(BaseDataset, self).__init__()
        self.args = args
        behaviors = pd.read_table(
            behaviors_path,
            header=None,
            names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        print(behaviors.head())
        behaviors.clicked_news.fillna(' ', inplace=True)
        behaviors.impressions = behaviors.impressions.str.split()
        
        if ratio is None:
            if epoch > 0:
                human_cnt, llm_cnt = 0, 0
                neg_human_cnt, neg_llm_cnt = 0, 0
                for row in tqdm(behaviors.itertuples()):
                    user_vector = user2vector[row.clicked_news].numpy()
                    impression_id = [impression.split('-')[0] for impression in row.impressions]
                    impression_label = [int(impression.split('-')[1]) for impression in row.impressions]
                    candidate_label = np.array(impression_label + impression_label)
                    human_news = [human_news2vector[id] for id in impression_id]
                    llm_news = [llm_news2vector[id] for id in impression_id]
                    if np.random.choice([1, 0], p=[0.5, 0.5]):
                        news = torch.stack(human_news + llm_news).numpy()
                        score = np.dot(news, user_vector)
                        select_pos_index = get_click_index(score, candidate_label, eta=args.eta)
                        index_list = list(range(len(score)))
                        if len(index_list) > args.negative_sampling_ratio:
                            select_neg_indexs = random.sample(index_list[:select_pos_index] + index_list[select_pos_index+1:], args.negative_sampling_ratio)
                        else:
                            select_neg_indexs = random.choices(index_list[:select_pos_index] + index_list[select_pos_index+1:], k=args.negative_sampling_ratio)
                        pos_news_id = impression_id[select_pos_index%len(impression_id)] + ('-human' if select_pos_index < len(impression_id) else '-llm')
                        neg_news_ids = [impression_id[select_neg_index%len(impression_id)] + ('-human' if select_neg_index < len(impression_id) else '-llm') for select_neg_index in select_neg_indexs]
                    else:
                        news = torch.stack(llm_news + human_news).numpy()
                        score = np.dot(news, user_vector)
                        select_pos_index = get_click_index(score, candidate_label, eta=args.eta)
                        index_list = list(range(len(score)))
                        if len(index_list) > args.negative_sampling_ratio:
                            select_neg_indexs = random.sample(index_list[:select_pos_index] + index_list[select_pos_index+1:], args.negative_sampling_ratio)
                        else:
                            select_neg_indexs = random.choices(index_list[:select_pos_index] + index_list[select_pos_index+1:], k=args.negative_sampling_ratio)
                        pos_news_id = impression_id[select_pos_index%len(impression_id)] + ('-llm' if select_pos_index < len(impression_id) else '-human')
                        neg_news_ids = [impression_id[select_neg_index%len(impression_id)] + ('-llm' if select_neg_index < len(impression_id) else '-human') for select_neg_index in select_neg_indexs]

                    if 'llm' in pos_news_id:
                        llm_cnt += 1
                    else:
                        human_cnt += 1

                    behaviors.at[row.Index, 'candidate_news'] = " ".join([pos_news_id] + neg_news_ids)
                new_llm_ratio = llm_cnt/(human_cnt + llm_cnt)
                for row in tqdm(behaviors.itertuples()):
                    behaviors.at[row.Index, 'clicked_news'] = " ".join([x+'-human' for x in row.clicked_news.split()])
            else:
                for row in tqdm(behaviors.itertuples()):
                    impression_id = [impression.split('-')[0] for impression in row.impressions]
                    impression_label = [int(impression.split('-')[1]) for impression in row.impressions]
                    pos_index = [i for i, impression in enumerate(row.impressions) if int(impression.split('-')[1]) == 1]
                    neg_index = [i for i, impression in enumerate(row.impressions) if int(impression.split('-')[1]) == 0]
                    select_pos_index = random.choice(pos_index)
                    if len(neg_index) < args.negative_sampling_ratio:
                        select_neg_indexs = random.choices(neg_index, k=args.negative_sampling_ratio)
                    else:
                        select_neg_indexs = random.sample(neg_index, args.negative_sampling_ratio)
                    pos_news_id = impression_id[select_pos_index] + '-human'
                    neg_news_ids = [impression_id[select_neg_index] + '-human' for select_neg_index in select_neg_indexs]
                    behaviors.at[row.Index, 'candidate_news'] = " ".join([pos_news_id] + neg_news_ids)
                    behaviors.at[row.Index, 'clicked_news'] = " ".join([x + '-human' for x in row.clicked_news.split()])
                new_llm_ratio, neg_llm_ratio = 0, 0
            self.new_ratio = new_llm_ratio
            print(f"\nnew ratio: {new_llm_ratio}")
        else:
            for row in tqdm(behaviors.itertuples()):
                impression_id = [impression.split('-')[0] for impression in row.impressions]
                impression_label = [int(impression.split('-')[1]) for impression in row.impressions]
                pos_index = [i for i, impression in enumerate(row.impressions) if int(impression.split('-')[1]) == 1]
                neg_index = [i for i, impression in enumerate(row.impressions) if int(impression.split('-')[1]) == 0]
                select_pos_index = random.choice(pos_index)
                if len(neg_index) < args.negative_sampling_ratio:
                    select_neg_indexs = random.choices(neg_index, k=args.negative_sampling_ratio)
                else:
                    select_neg_indexs = random.sample(neg_index, args.negative_sampling_ratio)
                pos_news_id = (impression_id[select_pos_index] + '-human') if np.random.choice([1, 0], p=[ratio, 1-ratio]) == 0 else (impression_id[select_pos_index] + '-llm')
                neg_news_ids = [impression_id[select_neg_index] + '-human' if np.random.choice([1, 0], p=[0.5, 0.5]) == 1 else impression_id[select_neg_index] + '-llm' for select_neg_index in select_neg_indexs]
                
                behaviors.at[row.Index, 'candidate_news'] = " ".join([pos_news_id] + neg_news_ids)
                behaviors.at[row.Index, 'clicked_news'] = " ".join([(x + '-human') if np.random.choice([1, 0], p=[ratio, 1-ratio]) == 0 else (x + '-llm') for x in row.clicked_news.split()])
            self.new_ratio = ratio
            print(f"\nnew ratio: {ratio}")
        self.behaviors_parsed = behaviors[["user", "clicked_news", "candidate_news"]]

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        item["clicked"] = 0
        impression_news = [x for x in row.candidate_news.split()]
        repeated_times = self.args.user_log_length - min(self.args.user_log_length, len(row.clicked_news.split()))
        assert repeated_times >= 0

        item["candidate_news_id"] = impression_news
        item["clicked_news_id"] = row.clicked_news.split()[-self.args.user_log_length:] + ['PADDED_NEWS'] * repeated_times
        item["log_mask"] = [1] * (self.args.user_log_length - repeated_times) + [0] * repeated_times
        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def collate_fn(batch):
            candidate_news_id = [i["candidate_news_id"] for i in batch]
            clicked_news_id = [i["clicked_news_id"] for i in batch]
            targets =  torch.stack([torch.tensor(i["clicked"]) for i in batch])
            log_mask = torch.stack([torch.tensor(i["log_mask"]) for i in batch])
            return {
                "candidate_news_id": candidate_news_id, #batch,
                "clicked_news_id": clicked_news_id,
                "targets": targets,
                "log_mask": log_mask
            }

        fn = collate_fn
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=fn)
    
class NewsDataset(Dataset):
    
    """
    Load news for evaluation.
    """
    def __init__(self, args, news_path):
        super(NewsDataset, self).__init__()
        self.args = args
        self.news_parsed = pd.read_csv(
            news_path,
            index_col='id',
            usecols=['id'] + use_cols, sep='\t')
        
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in tqdm(self.news2dict.keys()):
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    eval(self.news2dict[key1][key2]))

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        id = list(self.news2dict.keys())[idx]
        item = {"news": self.news2dict[id], "id": id}
        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def collate_fn(batch):

            return {
                "news": torch.stack([torch.cat(list(i["news"].values())) for i in batch]), # batch, num_words
                "id": [i["id"] for i in batch]
            }

        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=collate_fn)

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, args, behaviors_path, human_news2vector, llm_news2vector, ratio):
        super(UserDataset, self).__init__()
        self.args = args
        self.ratio = ratio
        self.human_news2vector = human_news2vector
        self.llm_news2vector = llm_news2vector
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.behaviors = pd.read_table(behaviors_path, header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        self.behaviors = self.behaviors[["user", "clicked_news", "impressions"]]
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        
        item = {
            "clicked_news": row.clicked_news.split()[-self.args.user_log_length:],
            "clicked_news_string": row.clicked_news, # 'N11 N22 N33': -> embedding
            "impressions": row.impressions
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = self.args.user_log_length - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] =  item["clicked_news"] + ['PADDED_NEWS'] * repeated_times

        item["log_mask"] = [1] * (self.args.user_log_length - repeated_times) + [0] * repeated_times
        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):
        
        def collate_fn(batch):
            candidate_news = [[c.split('-')[0] for c in i["impressions"] if c.split('-')[1] == '1'] for i in batch]
            return {
                "log_ids": torch.stack([torch.stack([self.human_news2vector[n] if np.random.choice([1,0], p=[self.ratio, 1-self.ratio]) == 0 else self.llm_news2vector[n] for n in i['clicked_news']]) for i in batch]),
                "log_mask": torch.stack([torch.tensor(i["log_mask"]) for i in batch]),
                "clicked_news_string": [i["clicked_news_string"] for i in batch],
                "candidate_news": candidate_news,
            }

        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=collate_fn)

class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, args, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.args = args
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def collate_fn(batch):
            impression_id = [i["impression_id"] for i in batch]
            candidate_news = [[c.split('-')[0] for c in i["impressions"]] for i in batch]
            clicked_news_string = [i["clicked_news_string"] for i in batch]
            y_true = [[int(c.split('-')[1]) for c in i["impressions"]] for i in batch]
            return {
                "impression_id": impression_id,
                "candidate_news": candidate_news, #batch,
                "clicked_news_string": clicked_news_string,
                "y_true": y_true,
            }

        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=collate_fn)

    def get_poscandidate_dataloader(self, batch_size, shuffle, num_workers):
        
        def collate_fn(batch):
            candidate_news = [[c.split('-')[0] for c in i["impressions"] if c.split('-')[1] == '1'] for i in batch]
            clicked_news_string = [i["clicked_news_string"] for i in batch]
            return {
                "candidate_news": candidate_news, #batch,
                "clicked_news_string": clicked_news_string,
            }
            
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=collate_fn)
        
if __name__ == '__main__':
    pass
    