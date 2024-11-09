""" 
Constants for the causalign package. 
"""

import os 
import torch

CAUSALIGN_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
CITING_ID_COL = "citingpaperid"
CITED_ID_COL = "citedpaperid"
NEGATIVE_ID_COL = "negativepaperid"
CORPUS_ID_COL = "corpus_paper_id"

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')