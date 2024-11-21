""" 
Logic for generating and preprocessing the train/val/test datasets for the ACL 
citations abstract-matching task and for the sentiment prediction tasks. 
"""

import pandas as pd 
import numpy as np
import os
import sys
TOP_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
if TOP_DIR not in sys.path:
    sys.path.insert(0, TOP_DIR)
from causalign.constants import CAUSALIGN_DIR, CITING_ID_COL, CITED_ID_COL, NEGATIVE_ID_COL, CORPUS_ID_COL
from tqdm.auto import tqdm
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, AutoTokenizer

class IMDBDataset(Dataset):
    def __init__(self, reviews, targets, args):
        self.reviews = reviews
        self.target = targets
        self.p = args
        self.max_length = args.max_seq_length

        self.tokenizer = None
        if args.pretrained_model_name in ['bert-base-uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        elif args.pretrained_model_name == 'msmarco-distilbert-base-v3':
            # Tokenizer initialization is not necessary since SentenceTransformer handles it
            self.tokenizer = DistilBertTokenizer.from_pretrained(f"sentence-transformers/{args.pretrained_model_name}")
        else:
            raise ValueError(f"Model {args.pretrained_model_name} not supported. Tokenizer could not be initialized.")
        
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.target[idx]

        encoding = self.tokenizer(review, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False, max_length=self.max_length)

        output = {
            'review_text': review,
            'target': target,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

        return output
    
class CivilCommentsDataset(Dataset):
    def __init__(self, text, toxicity, args):
        self.text = text
        self.toxicity = toxicity
        self.p = args
        self.max_length = args.max_seq_length

        self.tokenizer = None
        if args.pretrained_model_name in ['bert-base-uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        elif args.pretrained_model_name == 'msmarco-distilbert-base-v3':
            # Tokenizer initialization is not necessary since SentenceTransformer handles it
            self.tokenizer = DistilBertTokenizer.from_pretrained(f"sentence-transformers/{args.pretrained_model_name}")
        else:
            raise ValueError(f"Model {args.pretrained_model_name} not supported. Tokenizer could not be initialized.")
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        toxicity = self.toxicity[idx]

        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False, max_length=self.max_length)

        output = {
            'text': text,
            'toxicity': toxicity,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

        return output

class TextAlignDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = None
        if args.pretrained_model_name in ['bert-base-uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        elif args.pretrained_model_name == 'msmarco-distilbert-base-v3':
            # Tokenizer initialization is not necessary since SentenceTransformer handles it
            self.tokenizer = DistilBertTokenizer.from_pretrained(f"sentence-transformers/{args.pretrained_model_name}")
        else:
            raise ValueError(f"Model {args.pretrained_model_name} not supported. Tokenizer could not be initialized.")
        self.max_length = args.max_seq_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents_1 = [x[0] for x in data]
        sents_2 = [x[1] for x in data]
        sents_3 = [x[2] for x in data]

        # Tokenize and pad the sentences
        encoding1 = self.tokenizer(sents_1, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        encoding2 = self.tokenizer(sents_2, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        encoding3 = self.tokenizer(sents_3, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        return encoding1, encoding2, encoding3

    def collate_fn(self, batch):
        encoding1, encoding2, encoding3 = self.pad_data(batch)

        batched_data = {
            'input_ids_1': encoding1['input_ids'],
            'attention_mask_1': encoding1['attention_mask'],
            'token_type_ids_1': encoding1.get('token_type_ids'),  # Some models may not use token_type_ids
            'input_ids_2': encoding2['input_ids'],
            'attention_mask_2': encoding2['attention_mask'],
            'token_type_ids_2': encoding2.get('token_type_ids'),
            'input_ids_3': encoding3['input_ids'],
            'attention_mask_3': encoding3['attention_mask'],
            'token_type_ids_3': encoding3.get('token_type_ids')
        }

        # Convert token_type_ids to zeros if not present (to prevent potential errors)
        for key in ['token_type_ids_1', 'token_type_ids_2', 'token_type_ids_3']:
            if batched_data[key] is None:
                batched_data[key] = torch.zeros_like(batched_data['input_ids_1'])

        return batched_data


def load_acl_data(citation_file = 'acl_full_citations.parquet', 
        pub_info_file = 'acl-publication-info.74k.v2.parquet', 
        row_limit = None):
    data_directory = os.path.join(CAUSALIGN_DIR, 'data')

    print(f"Loading data from {data_directory}")
    df_cit = pd.read_parquet(os.path.join(data_directory, citation_file))
    df_pub = pd.read_parquet(os.path.join(data_directory, pub_info_file))

    dataset = create_triplets(df_cit, df_pub)
    
    if row_limit:
        dataset = dataset[:row_limit]

    return dataset

def create_triplets(df_cit, df_pub):
    # only keep data for ACL papers (otherwise merge will fail)
    df_cit_acl = df_cit[(df_cit['is_citedpaperid_acl'] == True) & (df_cit['is_citingpaperid_acl'] == True)].copy()

    # create a dictionary of citing papers as keys and all their cited papers as values
    citing_to_cited = df_cit_acl.groupby(CITING_ID_COL)[CITED_ID_COL].apply(list).to_dict()
    # get the list of all papers in the corpus
    all_corpus_papers = df_pub[CORPUS_ID_COL].unique()
    print("Creating negative labels...")
    tqdm.pandas(desc="Sampling negative examples...")
    df_cit_acl[NEGATIVE_ID_COL] = df_cit_acl.progress_apply(
                                        lambda x: add_negative_label(x, 
                                            citing_to_cited = citing_to_cited,
                                            all_papers = all_corpus_papers), 
                                        axis = 1)
    merged = df_cit_acl[[CITING_ID_COL, CITED_ID_COL, NEGATIVE_ID_COL]].merge( #get the citing abstract
        df_pub[[CORPUS_ID_COL, 'abstract']].rename(columns={'abstract': 'citing_abstract'}),
        left_on=CITING_ID_COL, right_on=CORPUS_ID_COL, how='inner'
    )
    merged = merged.merge(  # get the cited abstract
        df_pub[[CORPUS_ID_COL, 'abstract']].rename(columns={'abstract': 'cited_abstract'}),
        left_on=CITED_ID_COL, right_on=CORPUS_ID_COL, how='inner'
    )
    merged = merged.merge(  # get the negative abstract
        df_pub[[CORPUS_ID_COL, 'abstract']].rename(columns={'abstract': 'negative_abstract'}),
        left_on=NEGATIVE_ID_COL, right_on=CORPUS_ID_COL, how='inner'
    )        
    abstract_cols = ['citing_abstract', 'cited_abstract', 'negative_abstract']
    merged = merged[abstract_cols]
    merged = merged.dropna()
    triplets = list(merged.itertuples(index=False, name=None))

    return triplets


def add_negative_label(row,
        citing_to_cited: Dict[str, List], 
        all_papers: List[str]) -> pd.DataFrame:
    """
    Add negative examples to the positive examples. For each positive example, sample
    1 negative example by randomly selecting a paper from the corpus, 
    and confirming that the paper was not cited by `citingpaperid` from 
    the positive example.
    """
    # sample num_neg_samples negative examples for each positive example
    cited_papers = citing_to_cited.get(row['citingpaperid'], [])
    if not cited_papers:
        raise ValueError(f"No cited papers for {row['citingpaperid']}. This is unexpected.")
    neg_cited_paper = np.random.choice(all_papers)
    while neg_cited_paper in cited_papers:   # resample if we sampled a paper that was cited by the citing paper
        neg_cited_paper = np.random.choice(all_papers)
                
    return neg_cited_paper
