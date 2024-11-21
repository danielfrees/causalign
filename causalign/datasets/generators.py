""" 
Logic for generating and preprocessing the train/val/test datasets for the ACL 
citations abstract-matching task. 
"""
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


class IMDBDataset(Dataset):
    def __init__(self, reviews, targets, args):
        self.reviews = reviews
        self.target = targets
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