import pandas as pd 
import numpy as np
import os
import sys
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from datasets import Dataset, load_dataset, concatenate_datasets
from causalsent.constants import CAUSALSENT_DIR, CITING_ID_COL, CITED_ID_COL, NEGATIVE_ID_COL, CORPUS_ID_COL, ACL_DATA_DIR


def load_acl_data(citation_file: str = 'acl_full_citations.parquet', 
        pub_info_file: str = 'acl-publication-info.74k.v2.parquet', 
        row_limit: int = None)->List[Tuple]:
    data_directory = os.path.join(CAUSALSENT_DIR, ACL_DATA_DIR)

    print(f"Loading data from {data_directory}")
    df_cit = pd.read_parquet(os.path.join(data_directory, citation_file))
    df_pub = pd.read_parquet(os.path.join(data_directory, pub_info_file))

    dataset = create_triplets(df_cit, df_pub)
    
    if row_limit:
        dataset = dataset[:row_limit]

    return dataset

def create_triplets(df_cit: pd.DataFrame, df_pub: pd.DataFrame) -> List:
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


def load_imdb_data(split: str,
                imdb_data_source: str  = "stanfordnlp/imdb")->Dataset:
    imdb_ds = load_dataset(imdb_data_source, split = split)
    return imdb_ds

def load_civil_comments_data(split: str,
                civil_comments_data_source: str = "google/civil_comments", 
                binarize: bool = True,
                balance: bool = True):
    civil_ds = load_dataset(civil_comments_data_source, split=split)
    if binarize: 
        # map to binary toxicity by toxicity > 0. Heuristic, but makes sense
        # based on the rare distn of non-zeros. Unclear what Bansal did. 
        # They do resample to make even + subsample around their heuristic treatment words (kill)
        civil_ds = civil_ds.map(lambda x: {'toxicity': 1 if x['toxicity'] > 0 else 0})
    
    if balance:
        # subsample to balance classes
        civil_ds = civil_ds.shuffle(seed=328)
        num_toxic = civil_ds.filter(lambda x: x['toxicity'] == 1).num_rows
        num_nontoxic = civil_ds.filter(lambda x: x['toxicity'] == 0).num_rows
        min_class_size = min(num_toxic, num_nontoxic)
        
        toxic_ds = civil_ds.filter(lambda x: x['toxicity'] == 1).select(range(min_class_size))
        nontoxic_ds = civil_ds.filter(lambda x: x['toxicity'] == 0).select(range(min_class_size))

        # Concatenate the subsampled datasets using concatenate_datasets
        civil_ds = concatenate_datasets([toxic_ds, nontoxic_ds])
    
    return civil_ds