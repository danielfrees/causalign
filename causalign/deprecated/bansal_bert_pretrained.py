""" 
Implementation of our pre-trained BERT model. OG Bansal paper used: 'msmarco-distilbert-base-v3'
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import os
import transformers as ppb
from transformers import RobertaModel, AutoConfig
import warnings
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')
import copy
from transformers import get_scheduler,AdamW
from transformers import AutoTokenizer
import nltk
import scipy.sparse as sp
from constants import DEVICE


# OG Bansal model

class MSMarcoBERTModel(torch.nn.Module):
    def __init__(self,gamma):
        super().__init__()
        self.encoder = SentenceTransformer('msmarco-distilbert-base-v3')
        self.rep_dim = self.encoder.get_sentence_embedding_dimension()
        self.hidden_dim = 8192
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = torch.nn.Sequential(torch.nn.Linear(self.rep_dim,self.hidden_dim),
                torch.nn.BatchNorm1d(self.hidden_dim),
                torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                torch.nn.BatchNorm1d(self.hidden_dim),
                torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                torch.nn.BatchNorm1d(self.hidden_dim),
            #  torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.rep_dim))
        self.gamma = gamma
        
    def forward(self,x,y):
        device = self.predictor[0].weight.device
        x = self.to_gpu(x,device)
        y = self.to_gpu(y,device)
        # y_embeds = torch.nn.functional.normalize(self.predictor(torch.nn.functional.normalize(self.encoder(y)['sentence_embedding'],dim=-1)),dim=-1)
        # x_embeds = torch.nn.functional.normalize(self.target_encoder(x)['sentence_embedding'],dim=-1).clone().detach()
        flip = np.random.binomial(1,0.5)
        if (flip == 0):
            x_embeds = torch.nn.functional.normalize(self.predictor(torch.nn.functional.normalize(self.encoder(x)['sentence_embedding'],dim=-1)),dim=-1)
            # x_embeds = torch.nn.functional.normalize(self.predictor(self.encoder(x)['sentence_embedding']),dim=-1)
            y_embeds = torch.nn.functional.normalize(self.target_encoder(y)['sentence_embedding'],dim=-1).clone().detach()
        else:
            y_embeds = torch.nn.functional.normalize(self.predictor(torch.nn.functional.normalize(self.encoder(y)['sentence_embedding'],dim=-1)),dim=-1)
            # y_embeds = torch.nn.functional.normalize(self.predictor(self.encoder(y)['sentence_embedding']),dim=-1)
            x_embeds = torch.nn.functional.normalize(self.target_encoder(x)['sentence_embedding'],dim=-1).clone().detach()
        scores = x_embeds@y_embeds.T
        positives = torch.diag(scores).sum()
        negatives = scores.sum()-positives
        positives /= scores.shape[0]    # all items paired with themselves, avgd
        negatives /= ((scores.shape[0]-1)*(scores.shape[0]))  # avg over num of non-matching pairs
        self.update_target()
        return positives,negatives,-positives
    
    def get_embed(self,x):
        return torch.nn.functional.normalize(self.encoder(x)['sentence_embedding'],dim=-1)

    def to_gpu(self,x,device):
        return {'input_ids':x['input_ids'].to(device),'attention_mask':x['attention_mask'].to(device)}
    
    def update_target(self):
        target_dict = self.target_encoder.state_dict()
        online_dict = self.encoder.state_dict()
        for key in online_dict.keys():
            target_dict[key] = target_dict[key]*self.gamma + online_dict[key]*(1-self.gamma)
        self.target_encoder.load_state_dict(target_dict)