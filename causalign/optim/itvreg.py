""" 
Our implementation of Intervention-Based Regularization (ITVReg) algorithm. 
Proposed by Bansal et al. 2023. https://arxiv.org/pdf/2210.10636
"""

# TODO: Inherit from AdamW optimizer (I think thats what the original paper used as their base)
# TODO: Implement additional loss based on the difference in causal effects between the new encodings and base encodings
        # mask each individual token, 
        # compute dot product with the encoding without masking, 
        # abs value of the reduction in similarity, 
        # scale somehow
        
        # subtract causal effects for each token between new encodings and base encodings 
        # sum, normalize, scale by some lambda parameter 
        
        # add this loss to the adam loss 
        
        # I believe we also need to introduce some kind of momentum 
        # (gamma 0.995 momentum used in the OG implementation iirc)
