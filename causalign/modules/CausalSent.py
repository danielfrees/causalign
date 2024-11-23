import torch
from transformers import DistilBertModel

from heads import RieszHead, SentimentHead

class CausalSent(torch.nn.Module):
    def __init__(self,model_out_dim, pretrained_model_name):
        super().__init__()
        self.model_out_dim= model_out_dim

        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)

        self.reisz = RieszHead(model_out_dim)

        self.sentiment = SentimentHead(hidden_size=64, model_out_dim=model_out_dim, multilayer=False, probs=True)

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

            reisz_output_real = self.reisz(bert_output_real)
            reisz_output_treated = self.reisz(bert_output_treated)
            reisz_output_control = self.reisz(bert_output_control)

            reisz_loss = -2(reisz_output_treated - reisz_output_control) + (reisz_output_real ** 2) 

            sentiment_output_real = self.sentiment(bert_output_real)
            sentiment_output_treated = self.sentiment(bert_output_treated)
            sentiment_output_control = self.sentiment(bert_output_control)

            # Not sure if multiplying by label is correct. This will cause many tau_hats to be zero
            # tau_hat = reisz_output_real * label
            tau_hat = reisz_output_real * sentiment_output_real

            regularizer_loss = (sentiment_output_treated - sentiment_output_control - tau_hat) ** 2

            return sentiment_output_real, regularizer_loss, reisz_loss, 

        else:
            bert_output_real = self.bert(input_ids_real, attention_mask_real)

            sentiment_output_real = self.sentiment(bert_output_real)

            return sentiment_output_real, None, None

