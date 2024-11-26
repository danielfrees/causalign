import torch

class Lambda(torch.nn.Module):
    """
    A simple Lambda layer for inline tensor transformations. Needed for 
    dummy channel dimension in 'conv' layer.
    """
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def get_head(backbone_hidden_size: int,
            head_hidden_size: int,
            head_type: str,
            dropout_prob: float = 0.1) -> torch.nn.Module:
    """ 
    Produce a head for a given backbone hidden size and head type.
    
    Parameters:
    - backbone_hidden_size: int
        Size of the backbone's hidden output.
    - head_hidden_size: int
        Size of the hidden layers in the head.
    - head_type: str
        Type of head to construct ('fcn', 'conv', 'linear').
    - dropout_prob: float, default=0.1
        Dropout probability for regularization.
        
    Returns:
    - torch.nn.Module
        The constructed head.
    """
    if head_type == 'fcn':
        head = torch.nn.Sequential(
            torch.nn.Linear(backbone_hidden_size, head_hidden_size),
            torch.nn.LayerNorm(head_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(head_hidden_size, head_hidden_size // 2),
            torch.nn.LayerNorm(head_hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(head_hidden_size // 2, 1)
        )
    
    elif head_type == 'conv':
        head = torch.nn.Sequential(
            # Expect input shape: (batch_size, sequence_length, hidden_size)
            Lambda(lambda x: x.transpose(1, 2)),  # Transpose for Conv1d: (batch_size, hidden_size, sequence_length)
            torch.nn.Conv1d(backbone_hidden_size, head_hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(head_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(head_hidden_size, head_hidden_size // 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(head_hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),  # Pool to fixed size: (batch_size, hidden_size // 2, 1)
            torch.nn.Flatten(start_dim=1),  # Flatten to (batch_size, hidden_size // 2)
            torch.nn.Linear(head_hidden_size // 2, 1)  # Final prediction layer
        )
    elif head_type == 'linear':
        head = torch.nn.Sequential(torch.nn.Linear(backbone_hidden_size,1))
    else:
        raise ValueError(f"Invalid sentiment head type: {head_type}.")
    
    return head
class RieszHead(torch.nn.Module):
    def __init__(self, 
                backbone_hidden_size: int, 
                hidden_size: int,  
                head_type: str):
        super().__init__()
        self.backbone_hidden_size= backbone_hidden_size
        self.head = get_head(backbone_hidden_size, hidden_size, head_type)
        
    def forward(self, backbone_embedding):
        """ 
        Output Riesz Representers from the backbone embedding.
        Note that the embedding can be either pooled ('fcn' or 'linear' Riesz uses CLS or last-token pooled embedding), 
        or sequence ('conv' Riesz uses sequence of embeddings).
        """
        output = self.head(backbone_embedding)
        return output
    
class SentimentHead(torch.nn.Module):
    def __init__(self, 
                backbone_hidden_size: int, 
                hidden_size: int, 
                head_type: str,  
                probs=False):
        super().__init__()

        self.backbone_hidden_size= backbone_hidden_size
        self.probs = probs
        self.head = get_head(backbone_hidden_size, hidden_size, head_type)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, backbone_embedding):
        """ 
        Output sentiment predictions from the backbone embedding.
        Note that the embedding can be either pooled ('fcn' or 'linear' Riesz uses CLS or last-token pooled embedding), 
        or sequence ('conv' Riesz uses sequence of embeddings).
        """
        out = self.head(backbone_embedding)
        if self.probs: 
            out = self.sigmoid(out) # convert logits to probs optionally (! do not use during training)

        return out