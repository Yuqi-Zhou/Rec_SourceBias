import logging
import os
import sys
import torch
import numpy as np
import re
from metrics import fast_ndcg_score, fast_map_score

from model.BERT4Rec import BERT4Rec
from model.GRU4Rec import GRU4Rec
from model.SASRec import SASRec
from model.LRURec import LRURec

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    print(os.listdir(directory))
    if len(os.listdir(directory))==0:
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    print(f"ckpt_path: {ckpt_path}")
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None

class ScalarMovingAverage:

    def __init__(self, metric_type, eps=0):
        self.metric_type = metric_type
        self.avg_sum = 0
        self.avg_count = eps

    def add(self, value, len):
        self.avg_sum += value
        self.avg_count += len
        return self

    def get_avg_weight(self):
        return self.avg_sum / self.avg_count

class metrics:
    def __init__(self, name, ratio):
        self.name = name
        self.ratio = ratio
        
        self.NDCG1_eval = ScalarMovingAverage('NDCG1')
        self.NDCG3_eval = ScalarMovingAverage('NDCG3')
        self.NDCG5_eval = ScalarMovingAverage('NDCG5')
        
        self.MAP1_eval = ScalarMovingAverage('MAP1')
        self.MAP3_eval = ScalarMovingAverage('MAP3')
        self.MAP5_eval = ScalarMovingAverage('MAP5')
        
    
    def update(self, label, score):
        ndcg_results = fast_ndcg_score(label, score, k=[1,3,5])
        map_results = fast_map_score(label, score, k=[1,3,5])
        ndcg1 = ndcg_results[0]
        ndcg3 = ndcg_results[1]
        ndcg5 = ndcg_results[2]

        map1 = map_results[0]
        map3 = map_results[1]
        map5 = map_results[2]
        
        self.MAP1_eval.add(map1, 1)
        self.MAP3_eval.add(map3, 1)
        self.MAP5_eval.add(map5, 1)
    
        self.NDCG1_eval.add(ndcg1, 1)
        self.NDCG3_eval.add(ndcg3, 1)
        self.NDCG5_eval.add(ndcg5, 1)

    
    def print_result(self):
        logging.info("")
        logging.info("***"*20)
        logging.info(f"Test Ratio: {self.ratio}")
        logging.info(f"Test Type: {self.name}")
        logging.info("")
        logging.info(f"MAP1: {self.MAP1_eval.get_avg_weight()*100}")
        logging.info(f"MAP3: {self.MAP3_eval.get_avg_weight()*100}")
        logging.info(f"MAP5: {self.MAP5_eval.get_avg_weight()*100}")
        logging.info("")
        logging.info(f"nDCG1: {self.NDCG1_eval.get_avg_weight()*100}")
        logging.info(f"nDCG3: {self.NDCG3_eval.get_avg_weight()*100}")    
        logging.info(f"nDCG5: {self.NDCG5_eval.get_avg_weight()*100}")
        logging.info("***"*20)
        logging.info("")

def get_model(model_type):
    if model_type == "BERT4Rec":
        return BERT4Rec
    elif model_type == "GRU4Rec":
        return GRU4Rec
    elif model_type == "SASRec":
        return SASRec
    elif model_type == "LRURec":
        return LRURec
    else:
        return None

def get_mean(array):
    if len(array) == 0:
        return 0
    else:
        return np.mean(array).item()

def get_rate(array):
    a_len = len(array)
    return np.sum(array > 0)/a_len, np.sum(array < 0)/a_len, np.sum(array == 0)/a_len