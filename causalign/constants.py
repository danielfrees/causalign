""" 
Constants for the causalign package. 
"""

import os 
import torch
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set your Hugging Face token as an environment variable HF_TOKEN")

CAUSALIGN_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
CITING_ID_COL = "citingpaperid"
CITED_ID_COL = "citedpaperid"
NEGATIVE_ID_COL = "negativepaperid"
CORPUS_ID_COL = "corpus_paper_id"

SUPPORTED_BACKBONES_LIST = ["sentence-transformers/msmarco-distilbert-base-v3", 
                            'sentence-transformers/msmarco-distilbert-base-v4',
                            'meta-llama/Llama-3.1-8B']
DISTILBERT_SUPPORTED_MODELS = ["sentence-transformers/msmarco-distilbert-base-v3", 
                            'sentence-transformers/msmarco-distilbert-base-v4']

ACL_DATA_DIR = "acl_data"

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')