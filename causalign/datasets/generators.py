import pandas as pd 
import numpy as np
import os

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizerFast

class TextAlignDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent3 = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)
        encoding3 = self.tokenizer(sent3, return_tensors='pt', padding=True, truncation=True)

        token_ids1 = torch.LongTensor(encoding1['input_ids'])
        attention_mask1 = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids1 = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        token_ids3 = torch.LongTensor(encoding3['input_ids'])
        attention_mask3 = torch.LongTensor(encoding3['attention_mask'])
        token_type_ids3 = torch.LongTensor(encoding3['token_type_ids'])

        return (token_ids1, token_type_ids1, attention_mask1,
                token_ids2, token_type_ids2, attention_mask2,
                token_ids3, token_type_ids3, attention_mask3)

    def collate_fn(self, all_data):
        (token_ids1, token_type_ids1, attention_mask1,
         token_ids2, token_type_ids2, attention_mask2,
         token_ids3, token_type_ids3, attention_mask3) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids1,
                'token_type_ids_1': token_type_ids1,
                'attention_mask_1': attention_mask1,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'token_ids_3': token_ids3,
                'token_type_ids_3': token_type_ids3,
                'attention_mask_3': attention_mask3
            }

        return batched_data


def load_data():
    data_directory = os.path.dirname(os.path.abspath("__file__"))

    df_cit = pd.read_parquet(os.path.join(data_directory, 'acl_full_citations.parquet'))
    df_pub = pd.read_parquet(os.path.join(data_directory, 'acl-publication-info.74k.v2.parquet'))

    dataset = create_triplets(df_cit, df_pub)

    return dataset

def create_triplets(df_cit, df_pub):
    # only keep data for ACL papers (otherwise merge will fail)
    df_cit_acl = df_cit[(df_cit['is_citedpaperid_acl'] == True) & (df_cit['is_citingpaperid_acl'] == True)]

    df_cit_acl['negativepaperid'] = df_cit_acl.apply(add_negative_label)

    # create positive pairs
    # get the citing abstract
    merged = df_cit_acl[['citingpaperid', 'citedpaperid']].merge( #get the citing abstract
        df_pub[['corpus_paper_id', 'abstract']].rename(columns={'abstract': 'citing_abstract'}),
        left_on="citingpaperid", right_on="corpus_paper_id", how='inner'
    ).merge(  # get the cited abstract
        df_pub[['corpus_paper_id', 'abstract']].rename(columns={'abstract': 'cited_abstract'}),
        left_on="citedpaperid", right_on="corpus_paper_id", how='inner'
    ).merge(  # get the negative abstract
        df_pub[['corpus_paper_id', 'abstract']].rename(columns={'abstract': 'negative_abstract'}),
        left_on="negativepaperid", right_on="corpus_paper_id", how='inner'
    )    

    merged = merged[['citing_abstract', 'cited_abstract', 'negative_abstract']]

    triplets = list(merged.itertuples(index=False, name=None))

    return triplets


def add_negative_label(row,
                       df_pos: pd.DataFrame,
                       df_pub: pd.DataFrame) -> pd.DataFrame:
    """
    Add negative examples to the positive examples. For each positive example, sample
    `num_neg_samples` negative examples by randomly selecting a paper from the corpus, 
    and confirming that the paper was not cited by `citingpaperid` from 
    the positive example.
    """
    
    # create a dictionary of citing papers as keys and all their cited papers as values
    cited_papers = df_pos.loc[df_pos['citingpaperid']==row['citingpaperid'],'citedpaperid'].unique()
    
    # get the list of all papers in the corpus
    all_corpus_papers = df_pub['corpus_paper_id'].unique()
    
    # sample num_neg_samples negative examples for each positive example
    neg_cited_paper = np.random.choice(all_corpus_papers)
    while neg_cited_paper in cited_papers:   # resample if we sampled a paper that was cited by the citing paper
        neg_cited_paper = np.random.choice(all_corpus_papers)
                
    return neg_cited_paper