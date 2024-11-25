import random
import argparse
import numpy as np
import torch
import sqlite3
import os
import json

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def load_model_inference(filepath):
    return torch.load(filepath, weights_only=True)

def get_default_sent_training_args(regime: str):
    """ 
    Parse command line arguments for sentiment training.
    
    Params:
        regime: str. Options: 'causal_sent', 'intervention_sent'  # TODO: implement intervention_sent 
    """
    
    parser = argparse.ArgumentParser()

    if 'sent' in regime: 
        # ======= general training settings =======
        parser.add_argument("--pretrained_model_name", type=str, default="sentence-transformers/msmarco-distilbert-base-v4") 
        # type=str, default="bert-base-uncased")
        # "meta-llama/Llama-3.1-8B",
        
        # unfreeze either top{n} or all or iterative. each strings
        parser.add_argument("--unfreeze_backbone", type=str, default="all", help="Unfreeze the top{n} layers of the backbone model. Iterative unfreezes deeper throughout training. Options: 'top{n}', 'all', 'iterative'")
        parser.add_argument("--sentiment_head_type", type=str, default="fcn", help="Type of sentiment head to use. Options: 'fcn', 'linear', 'conv'")
        parser.add_argument("--riesz_head_type", type=str, default="fcn", help="Type of Riesz head to use. Options: 'fcn', 'linear', 'conv'")
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
        parser.add_argument("--limit_data", type=int, default=0, help='Number of rows to include from the dataset. Values <=0 do no subsetting.') #TODO: revert to full data
        parser.add_argument("--train_regime", type=str, default=regime, choices=['base', 'itvreg'])
        # early stopping
        parser.add_argument("--early_stop_patience", type=int, default=5, help='Number of epochs allowed with increased validation accuracy before stopping early.') #TODO: revert to full data
        parser.add_argument("--early_stop_delta", type=float, default=0.1, help='Amount of increased validation accuracy to tolerate in early stopping criterion.')
        
        # ======= data settings =======
        print("Selecting dataset for sentiment task (IMDB or CivilComments)...")
        # default data is loaded via huggingface. See dataset/utils.py for details for the IMDB, CivilComments data loading.
        parser.add_argument("--dataset", type=str, default="imdb") # choices=['imdb', 'civilcomments']
        parser.add_argument("--max_seq_length", type=int, default=100, help='Truncate texts to this number of tokens. Useful for faster training.')

        if regime == 'causal_sent':
            print("Setting hyperparameters for sentiment task...")
            # select the treatment word. Should be chosen heuristically s.t. it has meaning on the output.
            # The treatment word is the word upon which we will regularize the model 
            # to match riesz estimated word-level causal effects.
            # phrases may work, but not currently tested
            parser.add_argument("--treatment_phrase", type = str, default = "love", help = "The treatment word for the causal regularization regime.")
            parser.add_argument("--lambda_bce", type=float, default=1.0, help="Weight for the BCE loss term.")
            parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for the regularization loss term.")
            parser.add_argument("--lambda_riesz", type=float, default=1.0, help="Weight for the Riesz loss term.")
            parser.add_argument("--running_ate", action='store_true', default=False, help="Whether to track a running average or batch average to compute the Riesz regression ATE.")
            parser.add_argument("--estimate_targets_for_ate", action='store_true', default=False, help="Whether to use estimated sentiment probabilities or true targets to compute the Riesz regression ATE.")
        elif regime == 'intervention_sent':
            raise NotImplementedError("Intervention sentiment training not yet implemented.")
        else:
            raise ValueError(f"regime {regime} not recognized. Select one of 'causal_sent', 'intervention_sent'")
    else:
        raise ValueError(f"regime {regime} not recognized. Select one of 'causal_sent', 'intervention_sent'")
    
    # ignore unknown args such as when invoking from a notebook 
    args, unknown = parser.parse_known_args()
    return args

def get_default_sim_training_args(regime: str):
    """
    Parse command line arguments for the similarity training.
    """
    parser = argparse.ArgumentParser()

    if regime == 'base_sim':

        # ======= general training settings =======
        parser.add_argument("--pretrained_model_name", type=str, default="sentence-transformers/msmarco-distilbert-base-v4") 
        # type=str, default="bert-base-uncased")
        # "meta-llama/Llama-3.1-8B",
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
        parser.add_argument("--limit_data", type=int, default=0, help='Number of rows to include from the dataset. Values <=0 do no subsetting.') #TODO: revert to full data
        parser.add_argument("--train_regime", type=str, default=regime, choices=['base', 'itvreg'])
        # early stopping
        parser.add_argument("--early_stop_patience", type=int, default=5, help='Number of epochs allowed with increased validation accuracy before stopping early.') #TODO: revert to full data
        parser.add_argument("--early_stop_delta", type=float, default=0.1, help='Amount of increased validation accuracy to tolerate in early stopping criterion.')
        
    # ======= data settings =======
    print("Selecting  dataset for sentence similarity task (ACL)...")
    parser.add_argument("--acl_citation_filename", type=str, default="acl_full_citations.parquet")
    parser.add_argument("--acl_pub_info_filename", type=str, default="acl-publication-info.74k.v2.parquet")
    # parser.add_argument("--nli_filename", type=str, default="data/nli_for_simcse.csv") # general NLI case, code for this if you modify input data
    parser.add_argument("--max_seq_length", type=int, default=100, help='Truncate texts to this number of tokens. Useful for faster training.')

    print("Setting hyperparameters for sentence similarity task...")
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

class EarlyStopper:
    def __init__(self, 
                patience: int, 
                delta: float):
        self.count = 0
        self.patience = patience
        self.delta = delta
        self.max_val_acc = -1.0

    def highest_val_acc(self, val_acc: float):
        if val_acc > self.max_val_acc:
            self.max_val_acc = val_acc
            self.counter = 0
            return True
        
        return False

    def early_stop(self, val_acc: float):
        if val_acc < (self.max_val_acc - self.delta):
            self.counter += 1

            if self.counter >= self.patience:
                return True
        
        return False
    
# ======== Simple SQL DB for Experiment Tracking ========
def initialize_database(db_path="out/experiments.db"):
    """
    Initialize the SQLite database with tables for arguments, metrics, outputs, and model weights.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for arguments
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS arguments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create table for metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        train_acc REAL,
        train_f1 REAL,
        val_acc REAL,
        val_f1 REAL,
        test_acc REAL,
        test_f1 REAL,
        FOREIGN KEY (experiment_id) REFERENCES arguments (id)
    );
    """)

    # Create table for outputs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        split TEXT, -- train, val, test
        targets TEXT,
        predictions TEXT,
        FOREIGN KEY (experiment_id) REFERENCES arguments (id)
    );
    """)

    # Create table for model weights
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_weights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        weight_path TEXT,
        FOREIGN KEY (experiment_id) REFERENCES arguments (id)
    );
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


def save_arguments_to_db(args, db_path="out/experiments.db"):
    """
    Save argparse arguments into the 'arguments' table in the database.
    Dynamically adds columns for any argument keys not already present.
    
    Parameters:
        args: argparse.Namespace
            Arguments object containing all hyperparameters.
        db_path: str
            Path to the SQLite database file.
    Returns:
        experiment_id: int
            ID of the saved experiment row.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    args_dict = vars(args)  # Convert Namespace to dictionary
    
    # Get existing columns in the arguments table
    cursor.execute("PRAGMA table_info(arguments);")
    existing_columns = [row[1] for row in cursor.fetchall()]  # Column names are in the second field

    # Add missing columns
    for key in args_dict.keys():
        column_name = f"arg_{key}"
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE arguments ADD COLUMN {column_name} TEXT")

    # Insert values into the table
    columns = ", ".join([f"arg_{key}" for key in args_dict.keys()])
    placeholders = ", ".join(["?"] * len(args_dict))
    values = tuple(str(value) for value in args_dict.values())

    cursor.execute(f"INSERT INTO arguments ({columns}) VALUES ({placeholders})", values)
    experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"Arguments saved with Experiment ID: {experiment_id}")
    return experiment_id


def save_metrics_to_db(experiment_id, metrics, db_path="out/experiments.db"):
    """
    Save training, validation, and testing metrics into the 'metrics' table in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO metrics (experiment_id, train_acc, train_f1, val_acc, val_f1, test_acc, test_f1)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (experiment_id, metrics['train_acc'], metrics['train_f1'], metrics['val_acc'], 
        metrics['val_f1'], metrics['test_acc'], metrics['test_f1']))

    conn.commit()
    conn.close()
    print(f"Metrics saved for Experiment ID: {experiment_id}")


def save_outputs_to_db(experiment_id, split, targets, predictions, db_path="out/experiments.db"):
    """
    Save output targets and predictions into the 'outputs' table in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO outputs (experiment_id, split, targets, predictions)
    VALUES (?, ?, ?, ?)
    """, (experiment_id, split, json.dumps(targets), json.dumps(predictions)))

    conn.commit()
    conn.close()
    print(f"Outputs saved for {split} split in Experiment ID: {experiment_id}")


def save_model_weights_to_db(experiment_id, weight_path, db_path="out/experiments.db"):
    """
    Save the model weights' file path into the 'model_weights' table in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO model_weights (experiment_id, weight_path)
    VALUES (?, ?)
    """, (experiment_id, weight_path))

    conn.commit()
    conn.close()
    print(f"Model weights saved at {weight_path} for Experiment ID: {experiment_id}")
