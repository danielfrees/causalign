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
import warnings


def train_causal_sent(args):
    """ 
    Dataset preparation and training loop for the CausalSent model.
    """
    seed_everything(args.seed)
    
    # ====== Check Args ========
    if not args.interleave_training and args.lambda_reg > 0:
        warnings.warn("Regularization loss is enabled but interleave_training is disabled. Competing objectives will yield bad ATEs and bad model.")
        
    if args.interleave_training and not args.running_ate:
        raise ValueError("Interleaved training requires running_ate to be enabled. Pass --running_ate. We need a running ATE to compute the epoch ATEs for interleaved training.")
    
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
    
    # ======== Setup Interleaved Training if Req =========
    sentiment_epochs = None
    riesz_epochs = None
    if args.interleave_training:
        sentiment_epochs = {epoch: (epoch % 2 == 0) for epoch in range(epochs)}   # even epochs for sentiment, start w sentiment
        riesz_epochs = {epoch: (epoch % 2 != 0) for epoch in range(epochs)}   # odd epochs for riesz
    
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
        'trainable_backbone': model.percentage_trainable_backbone_params(),
        'trainable_model': model.percentage_trainable_params()
    }
    fraction_to_unfreeze: float = None
    args.unfreeze_backbone = process_unfreeze_param(args.unfreeze_backbone) 
    if args.unfreeze_backbone == "all" or isinstance(args.unfreeze_backbone, int):
        percent_trainable_params = model.unfreeze_backbone(num_layers = args.unfreeze_backbone)
    elif args.unfreeze_backbone == "iterative":
        fraction_to_unfreeze: float = 0  # iterates fractonally through training loop relative to epochs
    else:
        raise ValueError("unfreeze_backbone parsed badly: {args.unfreeze_backbone}")
    
    # Optimizer and BCE (sentiment) loss    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # ===== scale gradients for mixed precision training =====
    scaler = GradScaler() if args.autocast else None
    
    # track full epoch ATE estimates
    epoch_ate: float = None
    
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
        if args.unfreeze_backbone == "iterative":
            epoch_fraction = (epoch + 1) / epochs
            if epoch_fraction > fraction_to_unfreeze:
                fraction_to_unfreeze = epoch_fraction
                percent_trainable_params = model.unfreeze_backbone_fraction(fraction_to_unfreeze)  # only unfreeze when something changes (not a bug otherwise, just waste of time)
                
        # ========= Interleaved Training ==========
        training_sentiment: bool = False
        training_reg: bool = False
        training_riesz: bool = False
        if args.interleave_training:
            if sentiment_epochs[epoch]:
                print("[Interleaved Training] Training Sentiment Head")
                model.sentiment.requires_grad = True
                model.riesz.requires_grad = False
                
                lambda_riesz = 0 # no riesz loss when training sentiment head
                lambda_bce = args.lambda_bce
                lambda_reg = args.lambda_reg
                
                training_sentiment = True
                training_reg = True 

                if epoch == 0:  # don't regularize on the first epoch, we dont have a RR yet!
                    lambda_reg = 0
                    training_reg = False
                
            elif riesz_epochs[epoch]:
                print("[Interleaved Training] Training Riesz Head")
                model.sentiment.requires_grad = False
                model.riesz.requires_grad = True
                
                lambda_riesz = args.lambda_riesz
                lambda_bce = 0 # no sentiment loss when training riesz head
                lambda_reg = 0 # no regularization loss when training riesz head
                
                training_riesz = True
            else: 
                raise ValueError("Epoch not in sentiment or riesz epochs")
        
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
            treat_out = torch.sigmoid(sentiment_outputs_treated)
            control_out = torch.sigmoid(sentiment_outputs_control)
            real_out = torch.sigmoid(sentiment_outputs_real)
            
            # Compute tau_hat (estimated average treatment effect (ATE) of the 
            # selected treatment_phrase as estimated by a riesz representation
            # formula with RR computed via our simple implementation of RieszNet
            
            # note, whenever computing g (our estimate of the oracle sentiment analysis function)
            # we must apply sigmoid to the logits, otherwise treatment effects become nonsense 
            # in the binary sentiment domain 
            if not (args.interleave_training and training_sentiment):
                if running_ate:
                    # Compute batch-level numerator and denominator
                    batch_numer = None
                    batch_denom = None
                    if doubly_robust:
                        # TE_direct = g(X_i, 1) - g(X_i, 0), averaged later -> ATE_DIRECT
                        direct_te = treat_out - control_out
                        # TE_doublyrobust = TE_direct + RR(Z) * (Y - g(Z)), -> sum, -> averaged later by denom -> DR_ATE_DIRECT
                        batch_numer = torch.sum(direct_te + riesz_outputs_real * (targets - real_out))
                    else:
                        # RR(Z) * g(Z)  -- r.r. ATE, not doubly robust, averaged later
                        batch_numer = torch.sum(riesz_outputs_real * real_out)
                        
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
                        direct_ate = torch.mean(treat_out - control_out)
                        # ATE_doublyrobust = ATE_direct + E_n[RR(Z) * (Y - g(Z))]
                        tau_hat = direct_ate + torch.mean(riesz_outputs_real * (targets - real_out))
                    else:
                        # E_n[RR(Z) * g_0(Z)]  -- r.r. ATE, not doubly robust
                        tau_hat = torch.mean(riesz_outputs_real * real_out)
            elif training_reg: # if training sentiment head AND regularization, tau_hat should use the previous epoch_ate. Only when past the two warmup epochs and training reg
                tau_hat = torch.Tensor([epoch_ate]).to(device)
            else: # first sentiment epoch, no tau_hat yet and no regularization loss
                if not epoch == 0: 
                    raise ValueError("All epochs other than 0 should have a tau_hat. Something went wrong.")
                tau_hat = None
            
            if args.autocast:
                with autocast(device_type=str(device), dtype=torch.bfloat16):  # Use autocast for MPS
                    riesz_loss = 0 
                    reg_loss = 0
                    bce = 0
                    l1_loss = 0
                    if lambda_riesz > 0:
                        riesz_loss = torch.mean(-2 * (riesz_outputs_treated - riesz_outputs_control) + (riesz_outputs_real ** 2))
                    if lambda_reg > 0:
                        reg_loss = torch.mean(((treat_out - control_out) - tau_hat) ** 2)
                    if lambda_bce > 0:
                        bce = bce_loss(sentiment_outputs_real.squeeze(), targets)
                    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)    # L1 loss on trainable params
                    loss = lambda_bce * bce + lambda_reg * reg_loss + lambda_riesz * riesz_loss + lambda_l1 * l1_loss
            else:
                # Compute losses without autocast
                riesz_loss = 0 
                reg_loss = 0
                bce = 0
                l1_loss = 0
                if lambda_riesz > 0:
                    riesz_loss = torch.mean(-2 * (riesz_outputs_treated - riesz_outputs_control) + (riesz_outputs_real ** 2))
                if lambda_reg > 0:
                    reg_loss = torch.mean(((treat_out - control_out) - tau_hat) ** 2)
                if lambda_bce > 0:
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
                            f"Tau_Hat_{args.treatment_phrase}": tau_hat.item() if tau_hat else None,
                            "Batch": i + 1, 
                            "Backbone %Trainable": percent_trainable_params['trainable_backbone'],
                            "Model %Trainable": percent_trainable_params['trainable_model'],
                            "BCE Loss": bce.item() if bce else None,
                            "Reg Loss": reg_loss.item() if reg_loss else None,
                            "Riesz Loss": riesz_loss.item() if riesz_loss else None,
                            "L1 Loss": l1_loss.item() if l1_loss else None
                        }, 
                        )
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {train_acc:.4f}, "
                    f"F1: {train_f1:.4f}, "
                    f"Tau_Hat_{args.treatment_phrase}: {f'{tau_hat.item():.4f}' if tau_hat is not None else 'None'}, "
                    f"Backbone %Trainable: {percent_trainable_params['trainable_backbone']}, "
                    f"Model %Trainable: {percent_trainable_params['trainable_model']},"
                )

        # ====** end of epoch stuff **====
        
        # ====== Full Epoch ATE is the running ATE at end of Riesz epoch ======
        if training_riesz:
            epoch_ate = running_ate_numer / running_ate_denom
        
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
        wandb.log({"Val Accuracy": val_acc, "Val F1": val_f1, "Epoch": epoch + 1})
        if running_ate:
            wandb.log({f"Epoch_ATE_{args.treatment_phrase}": epoch_ate})
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
