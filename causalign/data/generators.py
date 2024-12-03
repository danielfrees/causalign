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
from typing import  List, Union
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, AutoTokenizer
import concurrent 
import warnings 
from causalign.constants import HF_TOKEN, DISTILBERT_SUPPORTED_MODELS

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

        if args.adjust_ate:
            dataset = create_synthetic_dataset(dataset=dataset, 
                                       fake_treatment_phrase=args.treatment_phrase,
                                       prop_treated=args.ate_change_treat_prop,
                                       diff_fake_ate=args.ate_change)

        self.limit_data = args.limit_data  # limit data for testing/ faster performance 
        if self.limit_data > 0:
            print(f"Limiting data to {self.limit_data} rows.")
            if len(dataset) < self.limit_data:
                warnings.warn(f"Dataset has only {len(dataset)} rows. Cannot limit to {self.limit_data}. Using full dataset.")
                sampled_indices = dataset
            else:
                sampled_indices = np.random.choice(len(dataset), self.limit_data, replace=False)
                dataset = dataset.select(sampled_indices)
        try:
            valid_data = [(text, label) for text, label in zip(dataset[text_col], dataset[label_col]) if text is not None]
            self.texts, self.targets = zip(*valid_data) if valid_data else ([], [])

            num_dropped = len(dataset[text_col]) - len(self.texts)

            if num_dropped > 0:
                warnings.warn(
                    f"Dataset {split}: Dropped {num_dropped} None or invalid texts. "
                    f"{len(self.texts)} valid texts remain."
                )
        except KeyError:
            raise ValueError(f"IMDB Dataset must contain {text_col} and {label_col} keys.")
        except TypeError:
            raise ValueError(f"Dataset contains invalid entries in column {text_col}.")
        self.p = args
        self.max_length = args.max_seq_length
        self.split = split
        self.treated_only = args.treated_only  # only use treated data for training (ATT instead of ATE)
        self.treatment_phrase = args.treatment_phrase  # treatment word for causal regularization

        if self.treated_only: # only include texts that include treatment phrase
            self.texts = [text for text in self.texts if self.treatment_phrase.lower() in text.lower()]
                        
        # ========= Tokenizer initialization =========
        self.tokenizer = None
        if args.pretrained_model_name in ['bert-base-uncased', 'meta-llama/Llama-3.1-8B']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, token = HF_TOKEN)
            if 'llama' in args.pretrained_model_name:
                self.tokenizer.pad_token = self.tokenizer.eos_token  #llama doesnt have a pad token
                print(f"Set llama pad token: {self.tokenizer.pad_token}")     
        elif args.pretrained_model_name in DISTILBERT_SUPPORTED_MODELS:
            # Tokenizer initialization is not necessary since SentenceTransformer handles it
            self.tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_model_name, token = HF_TOKEN)                
        else:
            raise ValueError(f"Model {args.pretrained_model_name} not supported. Tokenizer could not be initialized.")
        
        # ======== Create synthetic data for evaluation ========
        def create_synthetic_dataset(
                dataset: Dataset,
                fake_treatment_phrase: str = 'saucepan',
                prop_treated: float = '0.4',
                diff_fake_ate: float = '0.3',
                append_where: str = 'start',
                ignore_case: bool = True):
            
            if (prop_treated < 0) or (prop_treated > 1):
                raise ValueError(f"Please enter a synthetic proportion treated value between 0 and 1, given {prop_treated}")
            if (diff_fake_ate < 0) or (diff_fake_ate > 1):
                raise ValueError(f"Please enter a change in ATE between 0 and 1, given {diff_fake_ate}")
            
            n = len(dataset)
            # randomly determine treated rows
            treated_indices = np.random.choice(n, n*prop_treated, replace=False)

            # add fake treatment word to text
            dataset[treated_indices, text_col] = [treat_if_untreated(text, fake_treatment_phrase, append_where, ignore_case) for text in dataset[treated_indices, text_col]]

            # randomly flip outcomes to yield ATE of fake_ate
            treated_labels = dataset[treated_indices, label_col]
            indices_pos_label_treated = torch.nonzero(treated_labels == 1)
            indices_neg_label_treated = torch.nonzero(treated_labels == 0)
            
            prob_pos_label_given_treated = torch.mean(treated_labels[indices_pos_label_treated])
            prob_neg_label_given_treated = 1 - prob_pos_label_given_treated

            if (diff_fake_ate > prob_neg_label_given_treated) or  (diff_fake_ate < -prob_pos_label_given_treated):
                raise ValueError(f"Please enter a valid change in ATE, it must be between {-prob_pos_label_given_treated} and {prob_neg_label_given_treated}")

            switch_num = int(np.round(diff_fake_ate * len(treated_labels)))
            if diff_fake_ate > 0:
                switch_indices = np.random.choice(indices_neg_label_treated, switch_num, replace=False)

                treated_labels[switch_indices] = 1
            else:
                switch_indices = np.random.choice(indices_pos_label_treated, switch_num, replace=False)

                treated_labels[switch_indices] = 0

            dataset[treated_indices, label_col] = treated_labels

            return dataset

        
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
                warnings.warn(f"Appending treatment phrase '{treatment_phrase}' to the end of the text. BE SURE THAT YOU ARE NOT TRUNCATING TEXTS.")
                return text + ' ' + treatment_phrase
            elif append_where == 'start':
                return treatment_phrase + ' ' + text
            else:
                raise ValueError(f"append_where must be 'start' or 'end'. Got {append_where}.")

        def mask_if_present(
                text: str, 
                treatment_phrase: str, 
                tokenizer: Union[DistilBertTokenizer, AutoTokenizer], 
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
            mask_token = None
            if isinstance(tokenizer, DistilBertTokenizer):
                mask_token = tokenizer.mask_token
                #print(f"Tokenizer mask token: {mask_token}")
            else: 
                # llama is weird, need to use eos_token for mask
                if 'llama' in self.p.pretrained_model_name:
                    mask_token = tokenizer.pad_token
                    #print(f"Tokenizer mask token: {mask_token}")
                else:
                    raise ValueError("Mask token failed. Expected DistilBert or llama Autotokenizer.")
                # TODO: handle the above more elegantly
                
            if not mask_token:
                raise ValueError("Mask token failed. .eos_token or .mask_token not found in tokenizer.")
            
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
