import torch

class RieszHead(torch.nn.Module):
    def __init__(self, backbone_hidden_size: int):
        super().__init__()
        self.backbone_hidden_size= backbone_hidden_size
        self.linear = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,1))
        
    def forward(self, backbone_embedding):
        output = self.linear(backbone_embedding)
        return output
    
class SentimentHead(torch.nn.Module):
    def __init__(self, 
                hidden_size: int, 
                backbone_hidden_size: int, 
                head_type: str,  
                probs=False):
        super().__init__()

        self.backbone_hidden_size= backbone_hidden_size
        self.probs = probs

        if head_type == 'fcn':
            self.linear = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,hidden_size),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_size,1))
        else:
            self.linear = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,1))
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, backbone_embedding):
        out = self.linear(backbone_embedding)
        
        if self.probs: 
            out = self.sigmoid(out)

        return out