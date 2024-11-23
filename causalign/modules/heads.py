import torch
from transformers import DistilBertModel

class RieszHead(torch.nn.Module):
    def __init__(self,model_out_dim):
        super().__init__()
        self.model_out_dim= model_out_dim

        self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,1)).cuda()
        

    def forward(self, BERT_output):

        output = self.linear(BERT_output)

        return output
    
class SentimentHead(torch.nn.Module):
    def __init__(self,hidden_size, model_out_dim, multilayer=True, probs=False):
        super().__init__()

        self.model_out_dim= model_out_dim
        self.probs = probs

        if multilayer:
            self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,hidden_size),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_size,1)).cuda()
        else:
            self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,1)).cuda()
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,BERT_output):

        out = self.linear(BERT_output)

        out = self.sigmoid(out)

        return out