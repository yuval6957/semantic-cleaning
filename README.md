# semantic-cleaning

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

<a href="https://colab.research.google.com/github/yuval6957/semantic-cleaning/blob/main/nbs/index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This file will become your README and also the index of your
documentation.

## Install

``` sh
pip install semantic_cleaning
```

## How to use

``` python
import os
from tqdm.auto import tqdm
from typing import List, Dict, Set, Union, Callable
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F
import transformers
```

Processing a dataset to get a sentence for QA or comment and response
etc.

``` python
data = load_dataset("0-hero/OIG-small-chip2")
_ = preprocess_data(data,schema = ":{user} :{chip2}")
data['train']['_merged'][0]
```

    2

Compute the embadding fot the sentences

``` python
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to('cuda')
embedding = compute_embeddings(data = data, embedding_model = model, tokenizer = tokenizer, batch_size = 64, num_workers =16, dataset_feature  = '_merged'):
```

We can get the indicis of all the duplicated lines with the folowing
command:

``` python
to_delete = deduplicate_embeddings(embedded =embeddeing, epsilon=1e-2, batch_size=20000)
```

The full process could be run like this

``` python
  deduplicated = deduplicate_dataset(
      dataset = data['train'], 
      model = model, 
      tokenizer = tokenizer,
      epsilon = 1e-2, 
      model_batch_size = 64, 
      deduplication_batch_size = 20000, 
      num_workers = 16,
      dataset_feature = '_merged'
  )
  print (f"cleaned:{(1-len(deduplicated)/len(data['train']))*100:.2f}:%")
```

And deduplicated can be pushed back to the hub or saved on local drive
