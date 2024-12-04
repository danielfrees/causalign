import random
import argparse
import numpy as np
import torch
import sqlite3
import os
import json

def save_model(model, optimizer, args, filepath):
    """
    Save the model and all necessary components for reloading.

    Parameters:
    - model: torch.nn.Module
        The model to save.
    - optimizer: torch.optim.Optimizer
        The optimizer to save.
    - args: argparse.Namespace
        Training arguments or configuration to save.
    - filepath: str
        The path to save the model checkpoint.
    """
    save_info = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__,
        'pretrained_model_name': args.pretrained_model_name,  # Save the pretrained model name
        'sentiment_head_type': args.sentiment_head_type,      # Save the sentiment head type
        'riesz_head_type': args.riesz_head_type,              # Save the riesz head type
        'model_config': getattr(model, 'config', None),       # Save model config if available
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(save_info, filepath)
    print(f"Model saved to {filepath}")

def load_model_inference(filepath):
    """
    Load a model and its components for inference from a saved file.

    Parameters:
    - filepath: str
        The path to the saved model file.

    Returns:
    - model: torch.nn.Module
        The reloaded model with weights loaded.
    - args: argparse.Namespace
        Arguments saved with the model.
    """
    checkpoint = torch.load(filepath, weights_only=False)

    # Dynamically reconstruct the model
    model_class = checkpoint['model_class']
    pretrained_model_name = checkpoint.get('pretrained_model_name', None)
    sentiment_head_type = checkpoint.get('sentiment_head_type', 'fcn')
    riesz_head_type = checkpoint.get('riesz_head_type', 'fcn')

    if pretrained_model_name is None:
        raise ValueError("The checkpoint does not contain a 'pretrained_model_name' required to initialize the model.")

    # Reconstruct the model with all necessary architecture arguments
    model = model_class(pretrained_model_name, sentiment_head_type=sentiment_head_type,
                        riesz_head_type=riesz_head_type)
    model.load_state_dict(checkpoint['model_state_dict'])  # load best params

    print(f"Model loaded from {filepath}")
    return model, checkpoint['args']

def get_default_sent_training_args(regime: str):
    """ 
    Parse command line arguments for sentiment training.
    
    Params:
        regime: str. Options: 'causal_sent', 'intervention_sent'  # TODO: implement intervention_sent 
    """
    
    parser = argparse.ArgumentParser()

    if 'sent' in regime: 
        # ======= general training settings =======
        # project_name for namespacing wandb, sqlite tables, etc. 
        parser.add_argument("--project_name", type = str, default = "causal-sentiment")  #"causal-sentiment-architecture-doublyrobust-sigmoidfix"
        
        parser.add_argument("--pretrained_model_name", type=str, default="sentence-transformers/msmarco-distilbert-base-v4") 
        # type=str, default="bert-base-uncased")
        # "meta-llama/Llama-3.1-8B",
        
        # unfreeze either top{n} or all or iterative. each strings
        parser.add_argument("--unfreeze_backbone", type=str, default="all", help="Unfreeze the top{n} layers of the backbone model. Iterative unfreezes deeper throughout training. Options: 'top{n}', 'all', 'iterative'")
        parser.add_argument("--sentiment_head_type", type=str, default="fcn", help="Type of sentiment head to use. Options: 'fcn', 'linear', 'conv'")
        parser.add_argument("--riesz_head_type", type=str, default="fcn", help="Type of Riesz head to use. Options: 'fcn', 'linear', 'conv'")
        parser.add_argument("--seed", type=int, default=11711)
        parser.add_argument("--use_gpu", action='store_true', default = True)
        parser.add_argument("--autocast", action='store_true', default = False)  # mixed precision training, only good on cuda
        parser.add_argument("--num_workers", type=int, default=14)   # tune for your machine 
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')   # tune for your machine
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
        # logging 
        parser.add_argument("--log_every", type=int, default=5, help='Log training progress every n batches') # TODO: Revert to >200 for faster training on big dataset
        # limit data for testing 
        parser.add_argument("--limit_data", type=int, default=0, help='Number of rows to include from the dataset. Values <=0 do no subsetting.') #TODO: revert to full data
        parser.add_argument("--train_regime", type=str, default=regime, choices=['base', 'itvreg'])
        # early stopping
        parser.add_argument("--early_stop_patience", type=int, default=4, help='Number of epochs allowed with decreased validation accuracy before stopping early.') #TODO: revert to full data
        parser.add_argument("--early_stop_delta", type=float, default=0.05, help='Amount of decreased validation accuracy to tolerate in early stopping criterion.')
        
        # ======= data settings =======
        print("Selecting dataset for sentiment task (IMDB or CivilComments)...")
        # default data is loaded via huggingface. See dataset/utils.py for details for the IMDB, CivilComments data loading.
        parser.add_argument("--dataset", type=str, default="imdb") # choices=['imdb', 'civilcomments']
        parser.add_argument("--max_seq_length", type=int, default=100, help='Truncate texts to this number of tokens. Useful for faster training.')
        parser.add_argument("--treated_only", action='store_true', default=False, help="Whether to train only on treated samples. Then estimates an ATE instead of an ATT.")

        if regime == 'causal_sent':
            print("Setting hyperparameters for sentiment task...")
            # select the treatment word. Should be chosen heuristically s.t. it has meaning on the output.
            # The treatment word is the word upon which we will regularize the model 
            # to match riesz estimated word-level causal effects.
            # phrases may work, but not currently tested
            parser.add_argument("--treatment_phrase", type = str, default = "love", help = "The treatment word for the causal regularization regime.")
            parser.add_argument("--adjust_ate", action='store_true', default = False, help = "Determines whether the ATE is adjusted.")
            parser.add_argument("--synthetic_ate", type=float, default = 0.75, help = "If the ATE is to be adjusted, determines how much the difference will be.")
            parser.add_argument("--synthetic_ate_treat_fraction", type=float, default = 0.5, help = "If the ATE is to be adjusted, determines how big of a treated population should be created.")
            parser.add_argument("--lambda_bce", type=float, default=1.0, help="Weight for the BCE loss term.")
            parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for the regularization loss term.")
            parser.add_argument("--lambda_riesz", type=float, default=1.0, help="Weight for the Riesz loss term.")
            # recommend passing the following
            parser.add_argument("--running_ate", action='store_true', default=False, help="Whether to track a running average or batch average to compute the Riesz regression ATE.")
            parser.add_argument("--doubly_robust", action='store_true', default=False, help="Whether to use doubly robust estimation for the Riesz regression ATE.")
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
        parser.add_argument("--pretrained_model_name", type=str, default="c") 
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
    
def process_unfreeze_param(unfreeze_param: str):
    """
    Process the unfreeze parameter for the backbone model.
    """
    if unfreeze_param == 'all':
        return 'all'
    elif 'top' in unfreeze_param:
        return int(unfreeze_param.split('top')[-1])
    elif 'iterative' in unfreeze_param:
        return 'iterative'
    else:
        raise ValueError(f"Unfreeze parameter {unfreeze_param} not recognized. Select one of 'top[n]', 'all', 'iterative'")

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


def save_arguments_to_db(args, project_name, db_path="out/experiments.db"):
    """
    Save argparse arguments into the '<project_name>_arguments' table in the database.
    Dynamically adds columns for any argument keys not already present.

    Parameters:
        args: argparse.Namespace
            Arguments object containing all hyperparameters.
        project_name: str
            Name of the project to namespace tables.
        db_path: str
            Path to the SQLite database file.
    Returns:
        experiment_id: int
            ID of the saved experiment row.
    """
    table_name = f"{project_name}_arguments"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    args_dict = vars(args)  # Convert Namespace to dictionary

    # Get existing columns in the table
    cursor.execute(f"PRAGMA table_info({table_name});")
    existing_columns = [row[1] for row in cursor.fetchall()]  # Column names are in the second field

    # Add missing columns
    for key in args_dict.keys():
        column_name = f"arg_{key}"
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} TEXT")

    # Insert values into the table
    columns = ", ".join([f"arg_{key}" for key in args_dict.keys()])
    placeholders = ", ".join(["?"] * len(args_dict))
    values = tuple(str(value) for value in args_dict.values())

    cursor.execute(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values)
    experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"Arguments saved with Experiment ID: {experiment_id} in table: {table_name}")
    return experiment_id


def save_metrics_to_db(experiment_id, metrics, project_name, db_path="out/experiments.db"):
    """
    Save training, validation, and testing metrics into the '<project_name>_metrics' table in the database.
    """
    table_name = f"{project_name}_metrics"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        train_acc REAL,
        train_f1 REAL,
        val_acc REAL,
        val_f1 REAL,
        test_acc REAL,
        test_f1 REAL
    );
    """)

    cursor.execute(f"""
    INSERT INTO {table_name} (experiment_id, train_acc, train_f1, val_acc, val_f1, test_acc, test_f1)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (experiment_id, metrics['train_acc'], metrics['train_f1'], metrics['val_acc'], 
        metrics['val_f1'], metrics['test_acc'], metrics['test_f1']))

    conn.commit()
    conn.close()
    print(f"Metrics saved for Experiment ID: {experiment_id} in table: {table_name}")


def convert_to_native(obj):
    """
    Recursively converts NumPy types to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    else:
        return obj


def save_outputs_to_db(experiment_id, split, targets, predictions, project_name, db_path="out/experiments.db"):
    """
    Save output targets and predictions into the '<project_name>_outputs' table in the database.
    """
    table_name = f"{project_name}_outputs"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        split TEXT,
        targets TEXT,
        predictions TEXT
    );
    """)

    targets = convert_to_native(targets)
    predictions = convert_to_native(predictions)

    cursor.execute(f"""
    INSERT INTO {table_name} (experiment_id, split, targets, predictions)
    VALUES (?, ?, ?, ?)
    """, (experiment_id, split, json.dumps(targets), json.dumps(predictions)))

    conn.commit()
    conn.close()
    print(f"Outputs saved for {split} split in Experiment ID: {experiment_id} in table: {table_name}")


def save_model_weights_to_db(experiment_id, weight_path, project_name, db_path="out/experiments.db"):
    """
    Save the model weights' file path into the '<project_name>_model_weights' table in the database.
    """
    table_name = f"{project_name}_model_weights"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        weight_path TEXT
    );
    """)

    cursor.execute(f"""
    INSERT INTO {table_name} (experiment_id, weight_path)
    VALUES (?, ?)
    """, (experiment_id, weight_path))

    conn.commit()
    conn.close()
    print(f"Model weights saved at {weight_path} for Experiment ID: {experiment_id} in table: {table_name}")
