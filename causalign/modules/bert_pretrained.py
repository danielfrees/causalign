from torch import nn
from transformers import BertModel

class BertPreTrained(nn.Module):
    """
    BERT embedding model for extracting [CLS] token embeddings.
    """
    def __init__(self):
        super(BertPreTrained, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        pooler_output = bert_output['pooler_output']
        return pooler_output