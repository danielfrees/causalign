""" 
`bert_pretrained.py` contains the distilled BERT model used for CausAlign modeling.

Frozen encoder serves as the pre-trained reference for word 'causal effect' 
regularization techniques. Non-frozen encoder trained as normal. 
"""

from torch import nn
from transformers import DistilBertModel
from constants import DEVICE
from typing import Dict, Union
import torch

class SimDistilBERT(nn.Module):
    """
    DistilBERT embedding model for extracting embeddings from DistilBERT.
    """
    def __init__(self,
                args,
                pretrained_model_name='sentence-transformers/msmarco-distilbert-base-v3', 
                device=DEVICE):
        super(SimDistilBERT, self).__init__()
        self.p = args
        self.device = device

        # Layers: encoder, frozen encoder, logistic regression head
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.frozen_bert = DistilBertModel.from_pretrained(pretrained_model_name)

        # Set max sequence length from arguments
        self.bert.config.max_position_embeddings = args.max_seq_length
        self.frozen_bert.config.max_position_embeddings = args.max_seq_length
        self.logistic = nn.Sequential(
            nn.Linear(1, 1),  # TODO: Can look at more complex prediction head
            nn.Sigmoid()
        )

        # Train bert and logistic regression head, do not train frozen pre-trained encoder
        for param in self.bert.parameters():
            param.requires_grad = True
        for param in self.frozen_bert.parameters():
            param.requires_grad = False
        for param in self.logistic.parameters():
            param.requires_grad = True

        self.bert.to(self.device)
        self.frozen_bert.to(self.device)
        
        return None

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                frozen=False):
        """
        Encoding pass for DistilBERT.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs (batch size, seq length).
            attention_mask (torch.Tensor): Attention masks for input (batch size, seq length).
            frozen (bool): Whether to use the frozen encoder.

        Returns:
            torch.Tensor: The pooler output.
        """
        model = self.frozen_bert if frozen else self.bert
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        cls_embedding = hidden_state[:, 0, :]  # Output [CLS] token embeddings, captures sentence level info from self-attention mechanism
        
        return cls_embedding
    
    def forward(self, input_ids_1: torch.Tensor, attention_mask_1: torch.Tensor,
                    input_ids_2: torch.Tensor, attention_mask_2: torch.Tensor,
                    frozen = False):
        """ 
        Forward pass for SimDistilBERT.
        
        Params:
            input_ids_1: torch.Tensor, tokenized input IDs for sentence 1.
            attention_mask_1: torch.Tensor, attention mask for sentence 1.
            input_ids_2: torch.Tensor, tokenized input IDs for sentence 2.
            attention_mask_2: torch.Tensor, attention mask for sentence 2.
            frozen: bool, default=False. Whether to use the frozen encoder.
        Out:
            torch.Tensor[bool]: Labels for whether the sentences are similar. 
                1 for similar, 0 for not similar.
        """
        # Get embeddings for the two sentences
        emb_1 = self.encode(input_ids_1, attention_mask_1, frozen)
        emb_2 = self.encode(input_ids_2, attention_mask_2, frozen)
        
        # logistic regression head using cosine similarity
        entailed = self.logistic(torch.nn.functional.cosine_similarity(emb_1, emb_2).unsqueeze(1))

        return entailed
        
        