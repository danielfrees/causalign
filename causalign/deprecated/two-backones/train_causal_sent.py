""" 
Run Riesz Representer ATE estimation based causal effect regularized 
sentiment analysis. CausalSent. 
"""

import os
import sys
TOP_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
if TOP_DIR not in sys.path:
    sys.path.insert(0, TOP_DIR)
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score
from causalign.data.utils import load_imdb_data, load_civil_comments_data
from causalign.data.generators import IMDBDataset, CivilCommentsDataset
from causalign.modules.causal_sent import CausalSent
from causalign.data.generators import SimilarityDataset
from causalign.utils import (save_model, load_model_inference, 
                    get_default_sent_training_args, seed_everything, 
                    EarlyStopper, initialize_database, save_arguments_to_db,
                    save_metrics_to_db, save_model_weights_to_db, save_outputs_to_db,
                    process_unfreeze_param)
import wandb
import pandas as pd


def train_causal_sent(args):
    """ 
    Dataset preparation and training loop for the CausalSent model.
    """
    seed_everything(args.seed)
    
    project_name = args.project_name
    
    # ====== Verbose Argument Printout ======
    print("\n" + "="*50)
    print("Running CausalSent Training with the following arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50 + "\n")
    
    # ====== Initialize Experiment Tracking DB ======
    initialize_database()
    experiment_id = save_arguments_to_db(args=args, project_name=project_name)
    
    # ======= Setup Tracking and Device ========
    # Initialize wandb
    wandb.init(project=project_name, config=args)
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available() 
                        else "cpu")
    print(f"Using device: {device}")
    if str(device) == "mps" and args.autocast:
        raise ValueError("Mixed precision training not supported with MPS. Disable autocast.")
    
    # =========== Load Data ==============
    if args.dataset == "imdb":
        imdb_train_original = load_imdb_data(split = "train")
        imdb_train_splits = imdb_train_original.train_test_split(test_size=0.2)
        imdb_train = imdb_train_splits["train"]
        imdb_val = imdb_train_splits["test"]
        
    
        imdb_test = load_imdb_data(split = "test")

        imdb_ds_train: IMDBDataset = IMDBDataset(imdb_train, 
                                        split="train",
                                        args = args)
        imdb_ds_val: IMDBDataset = IMDBDataset(imdb_val,
                                            split = "validation", 
                                            args = args)
        imdb_ds_test: IMDBDataset = IMDBDataset(imdb_test,
                                            split = "test", 
                                            args = args)
        ds_train = imdb_ds_train
        ds_val = imdb_ds_val
        ds_test = imdb_ds_test
    else: 
        civil_train = load_civil_comments_data(split = "train")
        civil_val = load_civil_comments_data(split = "validation")
        civil_test = load_civil_comments_data(split = "test")
        
        civil_ds_train: CivilCommentsDataset = CivilCommentsDataset(civil_train, 
                                                    split="train",
                                                    args = args)
        civil_ds_val: CivilCommentsDataset = CivilCommentsDataset(civil_val,
                                                    split = "validation", 
                                                    args = args)
        civil_ds_test: CivilCommentsDataset = CivilCommentsDataset(civil_test,
                                                    split = "test", 
                                                    args = args)
        ds_train = civil_ds_train
        ds_val = civil_ds_val
        ds_test = civil_ds_test

    # ======== Setup Training ==========
    # Hyperparameters
    lambda_bce: float = args.lambda_bce
    lambda_reg: float = args.lambda_reg
    lambda_riesz: float = args.lambda_riesz
    lambda_l1: float = args.lambda_l1
    batch_size: int = args.batch_size
    epochs: int = args.epochs
    log_every: int = args.log_every
    running_ate: bool = args.running_ate # whether to track a running average or batch average to compute the RR ATE
    pretrained_model_name: str = args.pretrained_model_name
    lr: float = args.lr
    doubly_robust: bool = args.doubly_robust # whether to use doubly robust estimation of ATE
    
    # DataLoaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=SimilarityDataset.collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size, collate_fn=SimilarityDataset.collate_fn)
    test_loader = DataLoader(ds_test, batch_size=batch_size, collate_fn=SimilarityDataset.collate_fn)

    # Model, optimizer, and loss
    model = CausalSent(pretrained_model_name=pretrained_model_name, 
                    sentiment_head_type = args.sentiment_head_type, 
                    riesz_head_type = args.riesz_head_type).to(device)
    
    percent_trainable_params: dict = {
        'trainable_backbone_sentiment': model.percentage_trainable_component_params("backbone_sentiment"),
        'trainable_backbone_riesz': model.percentage_trainable_component_params("backbone_riesz"),
        'trainable_model': model.percentage_trainable_params()
    }
    
    # ==== setup backbone unfreezing =====
    fraction_to_unfreeze_sentiment: float = None
    fraction_to_unfreeze_riesz: float = None
    args.unfreeze_backbone_sentiment = process_unfreeze_param(args.unfreeze_backbone_sentiment) 
    if args.unfreeze_backbone_sentiment == "all" or isinstance(args.unfreeze_backbone_sentiment, int):
        percent_trainable_params = model.unfreeze_backbone(component = "backbone_sentiment", num_layers = args.unfreeze_backbone_sentiment)
    elif args.unfreeze_backbone == "iterative":
        fraction_to_unfreeze_sentiment: float = 0  # iterates fractonally through training loop relative to epochs
    else:
        raise ValueError(f"unfreeze_backbone_sentiment parsed badly: {args.unfreeze_backbone_sentiment}")
    args.unfreeze_backbone_riesz = process_unfreeze_param(args.unfreeze_backbone_riesz)
    if args.unfreeze_backbone_riesz == "all" or isinstance(args.unfreeze_backbone_riesz, int):
        percent_trainable_params = model.unfreeze_backbone(component = "backbone_riesz", num_layers = args.unfreeze_backbone_riesz)
    elif args.unfreeze_backbone == "iterative":
        fraction_to_unfreeze_riesz: float = 0
    else:
        raise ValueError(f"unfreeze_backbone_riesz parsed badly: {args.unfreeze_backbone_riesz}")
    
    # TODO: add in peft LoRA config stuff and pass to model for the backbone
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # 0.5 * lr every 5 epochs
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # ===== scale gradients for mixed precision training =====
    scaler = GradScaler() if args.autocast else None
    
    # ================ Training Loop =================
    early_stopper = EarlyStopper(patience=args.early_stop_patience, delta=args.early_stop_delta)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_targets, train_predictions = [], []
        
        # reset running ATE every epoch
        running_ate_numer: float = 0 
        running_ate_denom: float = 0
        
        # ========= Iterative Unfreezing ==========
        if args.unfreeze_backbone_sentiment == "iterative":
            epoch_fraction = (epoch + 1) / epochs
            if epoch_fraction > fraction_to_unfreeze_sentiment:
                fraction_to_unfreeze_sentiment = epoch_fraction
                percent_trainable_params = model.unfreeze_backbone_component_fraction(fraction = fraction_to_unfreeze_sentiment, 
                                                                component_str = "backbone_sentiment")  
        if args.unfreeze_backbone_riesz == "iterative":
            epoch_fraction = (epoch + 1) / epochs
            if epoch_fraction > fraction_to_unfreeze_riesz:
                fraction_to_unfreeze_riesz = epoch_fraction
                percent_trainable_params = model.unfreeze_backbone_component_fraction(fraction = fraction_to_unfreeze_riesz, 
                                                                component_str = "backbone_riesz")
        
        for i, batch in enumerate(train_loader):
            input_ids_real = batch['input_ids_real'].to(device)
            input_ids_treated = batch['input_ids_treated'].to(device)
            input_ids_control = batch['input_ids_control'].to(device)
            attention_mask_real = batch['attention_mask_real'].to(device)
            attention_mask_treated = batch['attention_mask_treated'].to(device)
            attention_mask_control = batch['attention_mask_control'].to(device)
            targets = batch['targets'].float().to(device)
            
            optimizer.zero_grad()  # Clear gradients

            # fwd pass
            (sentiment_outputs_real, sentiment_outputs_treated, sentiment_outputs_control, 
            riesz_outputs_real, riesz_outputs_treated, riesz_outputs_control) = model(
                input_ids_real,
                input_ids_treated,
                input_ids_control,
                attention_mask_real,
                attention_mask_treated,
                attention_mask_control,
            )

            # TODO: update this to use doubly robust as an option, 
            # remove direct targets option, it should be using the estimates 
            
            # Compute tau_hat (estimated average treatment effect (ATE) of the 
            # selected treatment_phrase as estimated by a riesz representation
            # formula with RR computed via our simple implementation of RieszNet
            
            # note, whenever computing g (our estimate of the oracle sentiment analysis function)
            # we must apply sigmoid to the logits, otherwise treatment effects become nonsense 
            # in the binary sentiment domain 
            if running_ate:
                # Compute batch-level numerator and denominator
                batch_numer = None
                batch_denom = None
                if doubly_robust:
                    # TE_direct = g(X_i, 1) - g(X_i, 0), averaged later -> ATE_DIRECT
                    direct_te = torch.sigmoid(sentiment_outputs_treated) - torch.sigmoid(sentiment_outputs_control)
                    # TE_doublyrobust = TE_direct + RR(Z) * (Y - g(Z)), -> sum, -> averaged later by denom -> DR_ATE_DIRECT
                    batch_numer = torch.sum(direct_te + riesz_outputs_real * (targets - torch.sigmoid(sentiment_outputs_real)))
                else:
                    # RR(Z) * g(Z)  -- r.r. ATE, not doubly robust, averaged later
                    batch_numer = torch.sum(riesz_outputs_real * torch.sigmoid(sentiment_outputs_real))
                    
                batch_denom = riesz_outputs_real.size(0)  # Batch size for E_n[.]

                # Update the running numerator and denominator
                running_ate_numer += batch_numer.item()
                running_ate_denom += batch_denom

                # Recompute tau_hat as the mean
                tau_hat = torch.Tensor([running_ate_numer / running_ate_denom]).to(device)
            else:
                tau_hat = None
                if doubly_robust:
                    # ATE_direct = E_n[g(X_i, 1), g(X_i, 0)]
                    direct_ate = torch.mean(torch.sigmoid(sentiment_outputs_treated) - torch.sigmoid(sentiment_outputs_control))
                    # ATE_doublyrobust = ATE_direct + E_n[RR(Z) * (Y - g(Z))]
                    tau_hat = direct_ate + torch.mean(riesz_outputs_real * (targets - torch.sigmoid(sentiment_outputs_real)))
                else:
                    # E_n[RR(Z) * g_0(Z)]  -- r.r. ATE, not doubly robust
                    tau_hat = torch.mean(riesz_outputs_real * torch.sigmoid(sentiment_outputs_real))
            
            if args.autocast:
                with autocast(device_type=str(device), dtype=torch.bfloat16):  # Use autocast for MPS
                    riesz_loss = torch.mean(-2 * (riesz_outputs_treated - riesz_outputs_control) + (riesz_outputs_real ** 2))
                    reg_loss = torch.mean(((sentiment_outputs_treated - sentiment_outputs_control) - tau_hat) ** 2)
                    bce = bce_loss(sentiment_outputs_real.squeeze(), targets)
                    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)    # L1 loss on trainable params
                    loss = lambda_bce * bce + lambda_reg * reg_loss + lambda_riesz * riesz_loss + lambda_l1 * l1_loss
            else:
                # Compute losses without autocast
                riesz_loss = torch.mean(-2 * (riesz_outputs_treated - riesz_outputs_control) + (riesz_outputs_real ** 2))
                reg_loss = torch.mean(((sentiment_outputs_treated - sentiment_outputs_control) - tau_hat) ** 2)
                bce = bce_loss(sentiment_outputs_real.squeeze(), targets)
                l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)    # L1 loss on trainable params
                loss = lambda_bce * bce + lambda_reg * reg_loss + lambda_riesz * riesz_loss + lambda_l1 * l1_loss

            if args.autocast:
                # scale gradients to avoid underflow/overflow if autocasting
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # =======   Logging   ========
            # Compute training metrics
            preds = torch.sigmoid(sentiment_outputs_real).squeeze().detach().cpu().numpy()
            # WHY IS THRESHOLD 0.5?
            preds = (preds > 0.5).astype(int)
            train_targets.extend(targets.cpu().numpy())
            train_predictions.extend(preds)

            if (i + 1) % log_every == 0:
                train_acc = accuracy_score(train_targets, train_predictions)
                train_f1 = f1_score(train_targets, train_predictions)
                wandb.log(
                        {"Train Loss": loss.item(), 
                            "Train Accuracy": train_acc, 
                            "Train F1": train_f1, 
                            f"Tau_Hat_{args.treatment_phrase}": tau_hat.item(),
                            "Batch": i + 1, 
                            "Sentiment Backbone %Trainable": percent_trainable_params['trainable_backbone_sentiment'],
                            "Riesz Backbone %Trainable": percent_trainable_params['trainable_backbone_riesz'],
                            "Model %Trainable": percent_trainable_params['trainable_model']
                        }, 
                        )
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {train_acc:.4f}, "
                    f"F1: {train_f1:.4f}, "
                    f"Tau_Hat_{args.treatment_phrase}: {tau_hat.item():.4f}, "
                    f"Sentiment Backbone %Trainable: {percent_trainable_params['trainable_backbone_sentiment']}, "
                    f"Riesz Backbone %Trainable: {percent_trainable_params['trainable_backbone_riesz']}, "
                    f"Model %Trainable: {percent_trainable_params['trainable_model']},"
                )
        # end of epoch 
        
        # ======= Scheduler Step =======
        scheduler.step()
        
        # ======= Validation Metrics (Log Every Epoch) =======
        model.eval()
        val_targets, val_predictions = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids_real = batch['input_ids_real'].to(device)
                attention_mask_real = batch['attention_mask_real'].to(device)
                targets = batch['targets'].float().to(device)
                
                sentiment_output_real = model(input_ids_real, None, None, attention_mask_real, None, None)
                preds = torch.sigmoid(sentiment_output_real).squeeze().cpu().numpy()
                preds = (preds > 0.5).astype(int)
                
                val_targets.extend(targets.cpu().numpy())
                val_predictions.extend(preds)
        
        # Compute validation metrics
        val_acc = accuracy_score(val_targets, val_predictions)
        val_f1 = f1_score(val_targets, val_predictions)
        wandb.log({"Val Accuracy": val_acc, 
                "Val F1": val_f1, 
                "Epoch": epoch + 1, 
                "LR": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch + 1}/{epochs} Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")  
        
        # ==== Early Stopping and Checkpointing ====
        if early_stopper.highest_val_acc(val_acc):
            model_path = os.path.join("out", f"experiment_{experiment_id}", "best_model.pt")
            save_model(model=model, optimizer=optimizer, args=args, filepath=model_path)
            save_model_weights_to_db(experiment_id=experiment_id, weight_path=model_path, project_name=project_name)
        if early_stopper.early_stop(val_acc):
            break
        # ==== end epoch ====
        
    best_model_path = os.path.join("out", f"experiment_{experiment_id}", "best_model.pt")
    best_model, _ = load_model_inference(best_model_path)
    best_model.eval()
    
    # compute outputs for full training, val, and test sets at the end and save
    # as csvs with verbose model name to out/
    train_targets, train_predictions = [], []
    with torch.no_grad():
        for batch in train_loader:
            input_ids_real = batch['input_ids_real'].to(device)
            attention_mask_real = batch['attention_mask_real'].to(device)
            targets = batch['targets'].float().to(device)
            
            sentiment_output_real = model(input_ids_real, None, None, attention_mask_real, None, None)
            preds = torch.sigmoid(sentiment_output_real).squeeze().cpu().numpy()
            preds = (preds > 0.5).astype(int)
            
            train_targets.extend(targets.cpu().numpy())
            train_predictions.extend(preds) 

    val_targets, val_predictions = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids_real = batch['input_ids_real'].to(device)
            attention_mask_real = batch['attention_mask_real'].to(device)
            targets = batch['targets'].float().to(device)
            
            sentiment_output_real = model(input_ids_real, None, None, attention_mask_real, None, None)
            preds = torch.sigmoid(sentiment_output_real).squeeze().cpu().numpy()
            preds = (preds > 0.5).astype(int)
            
            val_targets.extend(targets.cpu().numpy())
            val_predictions.extend(preds)

    test_targets, test_predictions = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids_real = batch['input_ids_real'].to(device)
            attention_mask_real = batch['attention_mask_real'].to(device)
            targets = batch['targets'].float().to(device)
            
            sentiment_output_real = model(input_ids_real, None, None, attention_mask_real, None, None)
            preds = torch.sigmoid(sentiment_output_real).squeeze().cpu().numpy()
            preds = (preds > 0.5).astype(int)
            
            test_targets.extend(targets.cpu().numpy())
            test_predictions.extend(preds)
    
    save_outputs_to_db(experiment_id=experiment_id, split="train", targets=train_targets, predictions=train_predictions, project_name=project_name)
    save_outputs_to_db(experiment_id=experiment_id, split="val", targets=val_targets, predictions=val_predictions, project_name=project_name)
    save_outputs_to_db(experiment_id=experiment_id, split="test", targets=test_targets, predictions=test_predictions, project_name=project_name)
    
    # save final metrics to out/
    train_acc = accuracy_score(train_targets, train_predictions)
    train_f1 = f1_score(train_targets, train_predictions)

    val_acc = accuracy_score(val_targets, val_predictions)
    val_f1 = f1_score(val_targets, val_predictions)

    test_acc = accuracy_score(test_targets, test_predictions)
    test_f1 = f1_score(test_targets, test_predictions)
    
    final_metrics = {
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    save_metrics_to_db(experiment_id=experiment_id, metrics=final_metrics, project_name=project_name)
    wandb.log({f"{k}_final": v for k, v in final_metrics.items()})

    return

        
if __name__ == "__main__":
    args = get_default_sent_training_args("causal_sent")
    train_causal_sent(args)
    print("Training complete :)")
