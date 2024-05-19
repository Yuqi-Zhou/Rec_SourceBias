import argparse
import utils
import torch
import random
import numpy as np
from pathlib import Path
from parameters import parse_args
from test import Tester
from train_loop import Trainer
from train_human import Trainer_main

import os
import logging

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    utils.setuplogger()
    args = parse_args()
    seed_torch(seed=args.seed)
    logging.info(f"args: {args}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if 'loop' == args.mode:
        trainer = Trainer(args)
        trainer.run()
    elif 'test' == args.mode:
        tester = Tester(args)
        tester.run()
    elif 'train' == args.mode:
        tester = Trainer_main(args)
        tester.run()
    
    