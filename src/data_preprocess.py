from os import path
from pathlib import Path
from tqdm import tqdm

import swifter
import utils
import logging
import numpy as np
import pandas as pd
from parameters import parse_args
from transformers import AutoTokenizer

utils.setuplogger()
args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
logging.info(f"args: {args}")
logging.info("Process data for training ")

def parse_news(source, target):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, word2int_path, entity2int_path: Path to load from
    """
    logging.info(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4],
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract'
                         ])
    news = news[['id', 'title', 'abstract']]
    news.fillna(' ', inplace=True)

    def parse_row(row):
        abstract = row['abstract'].lower()
        abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
            pad_to_max_length=True, truncation=True)
        if "bert-base-uncased" in args.plm_model:
            new_row = [
                row.id, abstract['input_ids'], abstract['token_type_ids'], abstract['attention_mask']
            ]
        elif "roberta-base" in args.plm_model:
            new_row = [
                row.id, abstract['input_ids'], [0] * len(abstract['input_ids']), abstract['attention_mask']
            ]
        return pd.Series(new_row,
                        index=[
                            'id', 'abstract_input_ids', 'abstract_token_type_ids', 'abstract_attention_mask'
                        ])

    parsed_news = news.swifter.apply(parse_row, axis=1)
    parsed_news.to_csv(target, sep='\t', index=False)

    

if __name__ == "__main__":
    parse_news(path.join(args.text_source),
               path.join(args.text_parsed_target))