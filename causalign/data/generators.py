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
from tqdm.auto import tqdm
from typing import  List
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, AutoTokenizer
import concurrent 

# ============ Helper functions for parallel tokenization ============
def tokenize_text(text, 
                    tokenizer: DistilBertTokenizer, 
                    args):
    """Function to tokenize a single review."""
    return tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        max_length=args.max_seq_length,
    )

def tokenize_texts(texts: List[str], 
                    tokenizer: DistilBertTokenizer, 
                    args):
    """ 
    Multi-threaded tokenization of texts.
    
    Args:
    - texts: List of review texts to tokenize.
    - tokenizer: Tokenizer object to use for encoding.
    
    Returns:
    - encodings: List of tokenized encodings for the texts. Each encoding will
    be a dictionary with keys 'input_ids' and 'attention_mask'.
    """
    max_threads = min(32, os.cpu_count() or 1)
    print(f"Truncating texts during tokenization to length: {args.max_seq_length}")
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        encodings = list(tqdm(
            executor.map(lambda x: tokenize_text(x, 
                                tokenizer = tokenizer, 
                                args= args), 
                        texts),
            desc="Tokenizing texts",
            total=len(texts),
        ))
                        
    return encodings
# =============================================================================     
        
class SimilarityDataset(Dataset):
    def __init__(self, dataset: Dataset, 
                    split: str, 
                    args, 
                    text_col: str,
                    label_col: str):
        """ 
        Expects to be initialized with the stanfordnlp/imdb dataset or google/civil_comments dataset. 
        
        Args:
        - dataset: Dataset object from Huggingface Datasets library.
        - split: Name of the split to use. Options are 'train', 'validation', 'test'.
        - args: Namespace object with training hyperparameters.
        - text_col: Name of the column containing the text data.
        - label_col: Name of the column containing the label data.
        """
        self.limit_data = args.limit_data  # limit data for testing/ faster performance 
        if self.limit_data > 0:
            print(f"Limiting data to {self.limit_data} rows.")
            sampled_indices = np.random.choice(len(dataset), self.limit_data, replace=False)
            dataset = dataset.select(sampled_indices)
        try:
            self.texts = dataset[text_col]
            self.targets = dataset[label_col]
        except KeyError:
            raise ValueError(f"IMDB Dataset must contain {text_col} and {label_col} keys.")
        self.p = args
        self.max_length = args.max_seq_length
        self.split = split
        self.treatment_phrase = args.treatment_phrase  # treatment word for causal regularization
        
        # ========= Tokenizer initialization =========
        self.tokenizer = None
        if args.pretrained_model_name in ['bert-base-uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        elif args.pretrained_model_name == "sentence-transformers/msmarco-distilbert-base-v3":
            # Tokenizer initialization is not necessary since SentenceTransformer handles it
            self.tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_model_name)
        else:
            raise ValueError(f"Model {args.pretrained_model_name} not supported. Tokenizer could not be initialized.")
        
        
        # ======== Create treated and control counterfactuals for each example ========
        def treat_if_untreated(
                text: str, 
                treatment_phrase: str, 
                append_where: str = 'start', 
                ignore_case: bool = True):
            """
            If the treatment_phrase is not present in the text, append it as 
            specified by append_where. Produces treatment counterfactuals.

            Args:
            - text: The text to treat.
            - treatment_phrase: The phrase to append to the text.
            - append_where: Where to append the treatment phrase. Options are 'end' or 'start'.
            - ignore_case: Whether to ignore case when checking for the treatment phrase.
            
            Returns:
            - treated_text: The treated text. If the text already contains the phrase, 
            the text is returned unchanged.
            """
            if ignore_case:
                if treatment_phrase.lower() in text.lower():
                    return text
            else:
                if treatment_phrase in text:
                    return text

            if append_where == 'end':
                return text + ' ' + treatment_phrase
            elif append_where == 'start':
                return treatment_phrase + ' ' + text
            else:
                raise ValueError(f"append_where must be 'start' or 'end'. Got {append_where}.")

        def mask_if_present(
                text: str, 
                treatment_phrase: str, 
                tokenizer: DistilBertTokenizer, 
                ignore_case: bool = True):
            """
            Mask the treatment phrase in the text if it exists. Produces control 
            counterfactuals. 

            Args:
            - text: The text to mask.
            - treatment_phrase: The phrase to mask.
            - tokenizer: The tokenizer to use for replacing the treatment phrase.
            - ignore_case: Whether to ignore case when replacing the treatment phrase.

            Returns:
            - control_text: The text with the treatment phrase replaced by '[MASK]'.
            """
            mask_token = tokenizer.mask_token
            if ignore_case:
                import re
                pattern = re.compile(re.escape(treatment_phrase), re.IGNORECASE)
                return pattern.sub(mask_token, text)
            else:
                return text.replace(treatment_phrase, mask_token)
        
        self.texts_treated = None
        self.texts_control = None
        if split == 'train':
            print("Creating treated and control counterfactuals...")
            self.texts_treated = [treat_if_untreated(text, self.treatment_phrase) for text in self.texts]
            self.texts_control = [mask_if_present(text, self.treatment_phrase, self.tokenizer) for text in self.texts]       
        # =============================================================================
        
        # ====== Produce encodings in parallel and cache =======
        print("Tokenizing texts for real, treated, and control counterfactuals...")
        self.encodings_real = tokenize_texts(texts = self.texts, 
                                        tokenizer = self.tokenizer, 
                                        args = args)
        if split == 'train':
            self.encodings_treated = tokenize_texts(texts = self.texts_treated,
                                                    tokenizer = self.tokenizer,
                                                    args = args)
            self.encodings_control = tokenize_texts(texts = self.texts_control,
                                                    tokenizer = self.tokenizer,
                                                    args = args)
            
    def __len__(self):
        return len(self.texts)

    def get_idx(self, idx):
        text = str(self.texts[idx])
        treated_text = str(self.texts_treated[idx] if self.texts_treated else None)
        control_text = str(self.texts_control[idx] if self.texts_control else None)
        target = self.targets[idx]

        encoding_real = self.encodings_real[idx]

        encoding_treated = None
        encoding_control = None
        if self.split == 'train':
            encoding_treated = self.encodings_treated[idx] if self.encodings_treated else None
            encoding_control = self.encodings_control[idx] if self.encodings_control else None
            
        output = {
            'text': text,
            'treated_text': treated_text,
            'control_text': control_text,
            'target': target,
            'input_ids_real': encoding_real['input_ids'],
            'input_ids_treated': encoding_treated.get('input_ids') if encoding_treated else None,
            'input_ids_control': encoding_control.get('input_ids') if encoding_control else None,
            'attention_mask_real': encoding_real['attention_mask'],
            'attention_mask_treated': encoding_treated.get('attention_mask') if encoding_treated else None,
            'attention_mask_control': encoding_control.get('attention_mask') if encoding_control else None
        }

        return output
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0: # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range.")
            return self.get_idx(idx = key) # get a single example
        else:
            raise TypeError(f"Invalid argument type: {type(key)}. Must be int or slice.")
        
    def collate_fn(batch):
        """Custom collate function for batching. Ensures equal padding for torch modules."""
        collated_data = {}

        for prefix in ['real', 'treated', 'control']:
            input_ids_key = f'input_ids_{prefix}'
            attention_mask_key = f'attention_mask_{prefix}'

            # Check if data for the current prefix exists
            if batch[0][input_ids_key] is not None:
                collated_data[input_ids_key] = torch.nn.utils.rnn.pad_sequence(
                    [item[input_ids_key].squeeze(0) for item in batch], batch_first=True
                )
                collated_data[attention_mask_key] = torch.nn.utils.rnn.pad_sequence(
                    [item[attention_mask_key].squeeze(0) for item in batch], batch_first=True
                )
            else:
                collated_data[input_ids_key] = None
                collated_data[attention_mask_key] = None

        # Targets should always be present
        collated_data['targets'] = torch.tensor([item['target'] for item in batch])

        return collated_data

class IMDBDataset(SimilarityDataset):
    def __init__(self, 
                dataset: Dataset, 
                split: str, 
                args):
        super().__init__(dataset, split, args, text_col = 'text', label_col = 'label')
    
class CivilCommentsDataset(SimilarityDataset):
    def __init__(self, 
                dataset: Dataset,
                split: str, 
                args):
        super().__init__(dataset, split, args, text_col = 'text', label_col = 'toxicity')


# ==== !!! note that the below is deprecated from original analysis and may not support newest training API/ args !!! ===
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
