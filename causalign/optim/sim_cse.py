""" 
Implementation of Contrastive Learning Loss.
"""

import torch
from torch import nn

class ContrastiveLearningLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastiveLearningLoss, self).__init__()
        self.tau = tau
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, premise_emb, entailment_emb, contradiction_emb):
        """
        premise_emb: tensor of size (batch_size, embedding_size)
        entailment_emb: tensor of size (batch_size, embedding_size)
        contradiction_emb: tensor of size (batch_size, embedding_size)
        """

        cos_sim_pos = self.cosine_similarity(premise_emb, entailment_emb) # ()
        cos_sim_neg = self.cosine_similarity(premise_emb, contradiction_emb)

        scaled_logit_pos = torch.exp(torch.div(cos_sim_pos, self.tau))
        scaled_logit_neg = torch.exp(torch.div(cos_sim_neg, self.tau))

        numerator = scaled_logit_pos
        denominator = scaled_logit_pos.sum(dim=0) + scaled_logit_neg.sum(dim=0)

        output = torch.mul(torch.log(torch.div(numerator, denominator)), -1)
        return output.mean()