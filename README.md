The implementation for SIGIR 2025: Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop.


## Quick Start

- For details of sequential recommendation models, please check the code in the folder `src/model`.

- For details of training, please check the file `src/train_human.py`

- For details of feebdack loop training, please check the code in the folder `src/train_loop.py`.

## File Structure
```shell
.
├── dataset  # * datset for training and testing
│   ├── Amazon_Beauty # * Beauty Dataset
│   ├── Amazon_Health # * Health Dataset
│   └── Amazon_Sports # * Sports Dataset
├── src
│   └── model # * popular sequential recommendation models
│       ├── BERT4Rec.py
│       ├── GRU4Rec.py
│       ├── LRURec.py
│       └── SASRec.py
│   ├── data_preprocess.py # * data preprocesss code
│   ├── Dataset.py # * Dataloader code
│   ├── metrics.py
│   ├── parameters.py # * training parameters
│   ├── run.py
│   ├── test.py
│   ├── train_human.py # * training code
│   ├── train_loop.py # * training in feedback loop code
│   └── utils.py
```

## Quick Example

```python
# data preprocess for tokenizeing text
$ python src/data_preprocess.py --plm_model plm/model/path \
    --text_source dataset/text/path \
    --text_parsed_target dataset/text/parsed/path

# train model in the HGC
$ python src/run.py --mode train \
    --model_type model_type --plm_model plm/model/path \
    --ckpt_dir save/model/path --human_news_file  HGC/text/path \
    --behaviors_file train/user/behavior/path

# test model in the item set mixed with HGC and AIGC
$ python src/run.py --mode test \
    --model_type model_type --plm_model plm/model/path \
    --human_news_file HGC/text/path \
    --llm_news_file AIGC/text/path \
    --behaviors_file  train/user/behavior/path \
    --test_behaviors_file test/user/behavior/path \
    --load_ckpt_name test/model/ckpt/path

# train model in the feedback loop
$ python src/run.py --mode loop \
    --model_type model_type --plm_model plm/model/path \
    --human_news_file HGC/text/path \
    --llm_news_file AIGC/text/path \
    --test_behaviors_file test/user/behavior/path \
    --behaviors_file1  loop/train/user/behavior/path/part1 \
    --behaviors_file2 loop/train/user/behavior/path/part2

# debais model in the feeback loop
$ python src/run.py --mode loop --debias \
    --model_type model_type --plm_model plm/model/path \
    --human_news_file HGC/text/path  \
    --llm_news_file AIGC/text/path \
    --llm_rewirte_news_file AIGC_rewritten/text/path \
    --test_behaviors_file test/user/behavior/path \
    --behaviors_file1 loop/train/user/behavior/path/part1 \
    --behaviors_file2 loop/train/user/behavior/path/part2
```

Note, please ensure that the `plm_model` and `news_file` correspond to each other and that the `behavior file` comes from the same domain.

## Dependencies

This repository has the following dependency requirements.

```
python==3.10.13
pandas==2.1.4
pytorch-pretrained-bert==0.6.2
pytorch-transformers==1.2.0
spacy==3.7.2
swifter==1.4.0
tensorflow==2.15.0.post1
tensorflow-estimator==2.15.0
tensorflow-gpu==2.9.0
torch==2.1.2+cu118
torchaudio==2.1.2+cu118
torchvision==0.16.2+cu118
transformers==4.39.3
```

The required packages can be installed via `pip install -r requirements.txt`.

## Citation

If you find our code or work useful for your research, please cite our work.

```
@article{zhou2024source,
  title={Source Echo Chamber: Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop},
  author={Zhou, Yuqi and Dai, Sunhao and Pang, Liang and Wang, Gang and Dong, Zhenhua and Xu, Jun and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2405.17998},
  year={2024}
}
```
