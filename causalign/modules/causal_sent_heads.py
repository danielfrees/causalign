import torch

class RieszHead(torch.nn.Module):
    def __init__(self,bert_hidden_size):
        super().__init__()
        self.bert_hidden_size= bert_hidden_size
        self.linear = torch.nn.Sequential(torch.nn.Linear(bert_hidden_size,1))
        
    def forward(self, bert_embedding):
        output = self.linear(bert_embedding)
        return output
    
class SentimentHead(torch.nn.Module):
    def __init__(self, 
                hidden_size, 
                bert_hidden_size, 
                multilayer=True, 
                probs=False):
        super().__init__()

        self.bert_hidden_size= bert_hidden_size
        self.probs = probs

        if multilayer:
            self.linear = torch.nn.Sequential(torch.nn.Linear(bert_hidden_size,hidden_size),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_size,1))
        else:
            self.linear = torch.nn.Sequential(torch.nn.Linear(bert_hidden_size,1))
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, bert_embedding):
        out = self.linear(bert_embedding)
        
        if self.probs: 
            out = self.sigmoid(out)

        return out