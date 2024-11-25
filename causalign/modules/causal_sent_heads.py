import torch

class RieszHead(torch.nn.Module):
    def __init__(self, 
                backbone_hidden_size: int, 
                head_type: str):
        super().__init__()
        self.backbone_hidden_size= backbone_hidden_size
        
        # TODO: implement diff head types
        self.head = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,1))
        
    def forward(self, backbone_embedding):
        output = self.head(backbone_embedding)
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

        # TODO: implement diff head types
        if head_type == 'fcn':
            self.head = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,hidden_size),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_size,1))
        else:
            self.head = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,1))
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, backbone_embedding):
        
        out = self.head(backbone_embedding)
        
        if self.probs: 
            out = self.sigmoid(out)

        return out