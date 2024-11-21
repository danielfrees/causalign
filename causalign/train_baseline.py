""" 
Run baseline training setup: SimCSE on base BERT encoder. 
"""
import os
import sys
import warnings 

TOP_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
if TOP_DIR not in sys.path:
    sys.path.insert(0, TOP_DIR)
from dotenv import load_dotenv
load_dotenv()

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler  
import wandb
wandb_key = os.getenv('WANDB_API_KEY')
if wandb_key:
    wandb.login(key=wandb_key)
else:
    warnings.warn("WANDB_API_KEY not found in environment variables. Please add to .env for logging.")


from tqdm import tqdm
from types import SimpleNamespace
import uuid


from datasets.generators import (
    TextAlignDataset,
    load_acl_data
)
from causalign.modules.bert_pretrained import SimDistilBERT
from causalign.optim.sim_cse import ContrastiveLearningLoss
from causalign.utils import save_model, get_training_args, seed_everything
from causalign.constants import DEVICE


TQDM_DISABLE = False


# Helper function for training loop. TODO: Refactorize this into a separate module when ITVReg is implemented.
def compute_loss(model, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, criterion, bce_loss_fn, 
                lambda_cse=1.0, lambda_entailment=1.0, lambda_contradiction=1.0):
    """
    Compute the combined loss for the SimCSE, entailment, and contradiction tasks.

    Parameters:
    - model: The model with encoding and entailment head functionality.
    - input_ids_1, attention_mask_1: Inputs for the premise sentences.
    - input_ids_2, attention_mask_2: Inputs for the entailment sentences.
    - input_ids_3, attention_mask_3: Inputs for the contradiction sentences.
    - criterion: The loss function for SimCSE (contrastive learning).
    - bce_loss_fn: Binary Cross-Entropy loss function for the entailment/contradiction classification.
    - lambda_cse, lambda_entailment, lambda_contradiction: Scalars to weight the respective losses.

    Returns:
    - total_loss: The weighted sum of the three losses.
    - entailment_logits, contradiction_logits: The predictions from the logistic regression head.
    - sim_p_e, sim_p_c: The average cosine similarities between the premise and entailment/contradiction embeddings.
    """
    # Encode embeddings using the model's encoder
    premise_emb = model.encode(input_ids_1, attention_mask_1)
    entailment_emb = model.encode(input_ids_2, attention_mask_2)
    contradiction_emb = model.encode(input_ids_3, attention_mask_3)

    # Compute the SimCSE loss
    loss_simcse = criterion(premise_emb, entailment_emb, contradiction_emb)
    
    # Compute the cosine similarities for logging
    sim_p_e = torch.nn.functional.cosine_similarity(premise_emb, entailment_emb)
    sim_p_c = torch.nn.functional.cosine_similarity(premise_emb, contradiction_emb)

    # Compute predictions through the entailment head
    entailment_logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
    contradiction_logits = model(input_ids_1, attention_mask_1, input_ids_3, attention_mask_3)

    # Targets for binary classification (entailments: 1, contradictions: 0)
    entailment_targets = torch.ones_like(entailment_logits)
    contradiction_targets = torch.zeros_like(contradiction_logits)

    # Compute the binary cross-entropy loss
    loss_entailment = bce_loss_fn(entailment_logits, entailment_targets)
    loss_contradiction = bce_loss_fn(contradiction_logits, contradiction_targets)

    # Weighted sum of the losses
    total_loss = lambda_cse * loss_simcse + lambda_entailment * loss_entailment + lambda_contradiction * loss_contradiction

    return total_loss, entailment_logits, contradiction_logits, sim_p_e, sim_p_c


def train_baseline(args):
    """
    Main function to run the training loop.
    """
    device = DEVICE if args.use_gpu else torch.device('cpu')
    print(f"Using Device: {device}")

    print("Loading ACL data...")
    nli_data = load_acl_data(citation_file=args.acl_citation_filename, 
                            pub_info_file=args.acl_pub_info_filename,
                            row_limit=args.limit_data)
    print("ACL Abstract Data loaded")
    dataset = TextAlignDataset(nli_data, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            collate_fn=dataset.collate_fn, num_workers=args.num_workers, pin_memory=True)
    print("ACL Abstract Dataloader created")

    # Setup config to be saved later, TODO: update this with more hyperparameters
    config = {'train_regime': args.train_regime,
            'data_dir': '.',
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'tau': args.tau,
            'max_seq_length': args.max_seq_length,
            'lambda_cse': args.lambda_cse,
            'lambda_entailment': args.lambda_entailment,
            'lambda_contradiction': args.lambda_contradiction,
            'row_limit': args.limit_data,
    }
    run_name = f"regime_{config['train_regime']}_id_{uuid.uuid4().hex[:8]}"
    if wandb.run is None:
        wandb.init(project='simcse-training', name=run_name, config=config)
    config = SimpleNamespace(**config)


    model = SimDistilBERT(args=args, device=device).to(device)
    print(f"Model initialized: {type(model).__name__}")    
    
    # SimCSE, BCELoss, AdamW for optimization of the base model 
    # SimCSE optimizes embedding space 
    # BCELoss optimizes the entailment and contradiction classification problem 
    # from the logistic regression head
    criterion = ContrastiveLearningLoss(tau=args.tau)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    model.train()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        running_loss = 0.0

        total_correct = 0
        total_examples = 0
        for i, batch in enumerate(tqdm(dataloader, disable=TQDM_DISABLE)):
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            input_ids_3 = batch['input_ids_3'].to(device)
            attention_mask_3 = batch['attention_mask_3'].to(device)

            optimizer.zero_grad()

            if args.autocast:
                with autocast():
                    total_loss, entailment_logits, contradiction_logits, sim_p_e, sim_p_c = compute_loss(
                        model, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                        input_ids_3, attention_mask_3, criterion, bce_loss_fn,
                        lambda_cse=args.lambda_cse, lambda_entailment=args.lambda_entailment, lambda_contradiction=args.lambda_contradiction
                    )

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss, entailment_logits, contradiction_logits, sim_p_e, sim_p_c = compute_loss(
                    model, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                    input_ids_3, attention_mask_3, criterion, bce_loss_fn,
                    lambda_cse=args.lambda_cse, lambda_entailment=args.lambda_entailment, lambda_contradiction=args.lambda_contradiction
                )

                total_loss.backward()
                optimizer.step()

            # Calculate correct predictions based on logistic regression outputs
            #TODO: fix accuracy calculation, something is wrong as i keep getting 0.5 constantly 
            correct_entailments = (entailment_logits > 0).float().sum().item()  # Positive logits imply entailment (true pos)
            correct_contradictions = (contradiction_logits <= 0).float().sum().item()  # Non-positive logits imply contradiction (true neg)
            batch_size = input_ids_1.size(0)
            total_batch_examples = batch_size * 2  # Two predictions per example: entailment + contradiction
            total_correct += correct_entailments + correct_contradictions
            total_examples += total_batch_examples

            running_loss += total_loss.item()
            
            # log metrics every log_every iterations
            if i % args.log_every == 0:
                avg_running_loss = running_loss / (i + 1)
                wandb.log({"train/loss": avg_running_loss, 
                        "train/accuracy": total_correct / total_examples,
                        "train/entailment_avg_sim": sim_p_e.mean().item(),
                        "train/contradiction_avg_sim": sim_p_c.mean().item(),
                        "iteration": i})

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
        print(f"Accuracy: {total_correct / total_examples}")

    # Save the trained model
    save_model(model, optimizer, args, config, args.output_model_path)
    print(f"Model saved to {args.output_model_path}")


if __name__ == '__main__':
    args = get_training_args(regime='base')
    args.output_model_path = f'params-{args.train_regime}-{args.epochs}-{args.lr}-{args.tau}-contrastive-baseline.pt'  # Save path.
    seed_everything(args.seed)
    train_baseline(args)