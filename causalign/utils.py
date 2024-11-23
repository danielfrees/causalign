import random
import argparse
import numpy as np
import torch

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def get_training_args(regime: str):
    """
    Parse command line arguments.
    
    Params:
        regime: str, default='base'. Options: 'base_sim', 'base_sent' #TODO: optimize for 'riesz' and 'itvreg' later
        Selects the default hyperparameters for the given regime. 
    """
    parser = argparse.ArgumentParser()

    if regime.startswith('base'):
        
        # ======= general training settings =======
        parser.add_argument("--seed", type=int, default=11711)
        parser.add_argument("--use_gpu", action='store_true', default = True)
        parser.add_argument("--autocast", action='store_true', default = torch.cuda.is_available())  # mixed precision training, only good on cuda
        parser.add_argument("--num_workers", type=int, default=14)   # tune for your machine 
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')   # tune for your machine
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
        # logging 
        parser.add_argument("--log_every", type=int, default=5, help='Log training progress every n batches') # TODO: Revert to >200 for faster training on big dataset
        # limit data for testing 
        parser.add_argument("--limit_data", type=int, default=500, help='Number of rows to include from the dataset.') #TODO: revert to full data
        parser.add_argument("--train_regime", type=str, default=regime, choices=['base', 'itvreg'])
        if regime == 'base_sim' or regime == 'base_sent':
            parser.add_argument("--pretrained_model_name", type=str, default="msmarco-distilbert-base-v3") # type=str, default="bert-base-uncased")
        else:
            raise ValueError(f"train_regime {regime} not recognized. Select one of 'base_sim', 'base_sent'")  
        
        # ======= data settings =======
        if regime == 'base_sim':
            print("Setting up hyperparameters for sentence similarity task (ACL)...")
            parser.add_argument("--acl_citation_filename", type=str, default="acl_full_citations.parquet")
            parser.add_argument("--acl_pub_info_filename", type=str, default="acl-publication-info.74k.v2.parquet")
            # parser.add_argument("--nli_filename", type=str, default="data/nli_for_simcse.csv") # general NLI case, code for this if you modify input data
        elif regime == 'base_sent':
            print("Setting up hyperparameters for sentiment task (IMDB, CivilComments)...")
            pass # default data is loaded via huggingface. See dataset/utils.py for details for the IMDB, CivilComments data loading.
        
        # ======= specific hyperparameters =======
        parser.add_argument("--max_seq_length", type=int, default=100, help='Truncate texts to this number of tokens. Useful for faster training.')

        if regime == 'base_sim':
            parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter for contrastive loss')
            parser.add_argument("--curriculum_training", action='store_true')
            parser.add_argument("--lambda_", type=float, default=1.0, help='lambda for pacing function')
            parser.add_argument("--distance_margin", type=float, default=0.2)
            parser.add_argument("--sort_by_cosine_entailment", action='store_true')
            parser.add_argument("--sort_by_cosine_contradiction", action='store_true')
            
            # weights for the total loss
            parser.add_argument("--lambda_cse", type=float, default=1.0, help='Weight for SimCSE loss')
            parser.add_argument("--lambda_entailment", type=float, default=0.25, help='Weight for entailment BCE loss')
            parser.add_argument("--lambda_contradiction", type=float, default=0.25, help='Weight for contradiction BCE loss')
        elif regime == 'base_sent':
            # select the treatment word. Should be chosen heuristically s.t. it has meaning on the output.
            # The treatment word is the word upon which we will regularize the model 
            # to match riesz estimated word-level causal effects.
            # phrases may work, but not currently tested
            parser.add_argument("--treatment_phrase", type = str, default = "love", help = "The treatment word for the causal regularization regime.")
    else: 
        raise ValueError(f"regime {regime} not recognized. Select one of 'base_sim', 'base_sent'")
    
    # ignore unknown args such as when invoking from a notebook 
    args, unknown = parser.parse_known_args()
    return args

def seed_everything(seed=11711):
    """
    Set random seed for all packages and functions
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True