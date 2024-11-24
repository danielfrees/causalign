import torch
from transformers import DistilBertModel

from causalign.modules.causal_sent_heads import RieszHead, SentimentHead

class CausalSent(torch.nn.Module):
    def __init__(self, 
                bert_hidden_size: int = 768, 
                pretrained_model_name: str = 'sentence-transformers/msmarco-distilbert-base-v3'):
        super().__init__()
        self.bert_hidden_size= bert_hidden_size
        
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.riesz = RieszHead(bert_hidden_size)
        self.sentiment = SentimentHead(hidden_size=64, bert_hidden_size=bert_hidden_size, multilayer=False, probs=False)

    def forward(self,
                input_ids_real, 
                input_ids_treated, 
                input_ids_control, 
                attention_mask_real, 
                attention_mask_treated, 
                attention_mask_control):
        
        if self.training:
            bert_output_real = self.bert(input_ids_real, attention_mask_real)
            bert_output_treated = self.bert(input_ids_treated, attention_mask_treated)
            bert_output_control = self.bert(input_ids_control, attention_mask_control)

            # Use BERT to produce `last_hidden_state`embedding
            hidden_state_real = bert_output_real.last_hidden_state
            hidden_state_treated = bert_output_treated.last_hidden_state
            hidden_state_control = bert_output_control.last_hidden_state

            # Use CLS token representation (first token of the sequence)
            # captures overall sentence information 
            cls_token_real = hidden_state_real[:, 0, :]
            cls_token_treated = hidden_state_treated[:, 0, :]
            cls_token_control = hidden_state_control[:, 0, :]

            riesz_output_real = self.riesz(bert_embedding = cls_token_real)
            riesz_output_treated = self.riesz(bert_embedding = cls_token_treated)
            riesz_output_control = self.riesz(bert_embedding = cls_token_control)

            sentiment_output_real = self.sentiment(bert_embedding = cls_token_real)
            sentiment_output_treated = self.sentiment(bert_embedding = cls_token_treated)
            sentiment_output_control = self.sentiment(bert_embedding = cls_token_control)

            return (sentiment_output_real, sentiment_output_treated, sentiment_output_control, 
                    riesz_output_real, riesz_output_treated, riesz_output_control)
        else:
            bert_output_real = self.bert(input_ids_real, attention_mask_real)
            cls_token_real = bert_output_real.last_hidden_state[:, 0, :]
            sentiment_output_real = self.sentiment(bert_embedding = cls_token_real)
            return sentiment_output_real

