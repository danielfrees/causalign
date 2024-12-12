from transformers import DistilBertModel, LlamaModel
import torch
from causalsent.modules.causal_sent_heads import RieszHead, SentimentHead
from causalsent.constants import SUPPORTED_BACKBONES_LIST, HF_TOKEN
from typing import Union
import warnings


class CausalSent(torch.nn.Module):
    def __init__(self, 
                pretrained_model_name: str,
                sentiment_head_type = 'fcn', # 'fcn', 'linear', 'conv'
                riesz_head_type = 'fcn', # 'fcn', 'linear', 'conv'
                ):
        """ 
        Causal Sentence Embedding Model.
        
        Parameters:
        - pretrained_model_name: str
            Pretrained model name for the backbone model.
        - sentiment_head_type: str, default='fcn'
            Type of sentiment head to use. Options: 'fcn', 'linear', 'conv'
        - riesz_head_type: str, default='fcn'
            Type of Riesz head to use. Options: 'fcn', 'linear', 'conv'
        """
        
        super().__init__()
        self.sentiment_head_type = sentiment_head_type
        self.riesz_head_type = riesz_head_type
        
        # =========== Load backbone (DistilBERT or LLaMA) =================
        if not pretrained_model_name in SUPPORTED_BACKBONES_LIST:
            warnings.warn(f"[WARNING] Unsupported/tested model name: {pretrained_model_name}. ")
            print(f"Supported models: {SUPPORTED_BACKBONES_LIST}")
            
        if "bert" in pretrained_model_name:
            self.backbone_riesz = DistilBertModel.from_pretrained(pretrained_model_name, token=HF_TOKEN)
            self.backbone_sentiment = DistilBertModel.from_pretrained(pretrained_model_name, token=HF_TOKEN)
            backbone_type = "DistilBERT"
        elif "llama" in pretrained_model_name:
            # Example: Uncomment below if LLaMA 3.1 8B is desired
            self.backbone_riesz = LlamaModel.from_pretrained(pretrained_model_name, token=HF_TOKEN)
            self.backbone_sentiment = LlamaModel.from_pretrained(pretrained_model_name, token=HF_TOKEN)
            backbone_type = "LLaMA"
        else:
            raise ValueError(f"[ERROR] Unsupported model name: {pretrained_model_name}. "
                            f"Expected 'bert' or 'llama' in the name.")
            
        # sentiment and riesz backbones always use the same model and share their hidden size
        backbone_hidden_size = self.backbone_sentiment.config.hidden_size 
        self.backbone_hidden_size = backbone_hidden_size

        # Freeze backbone parameters initially
        for param in self.backbone_riesz.parameters():
            param.requires_grad = False
        for param in self.backbone_sentiment.parameters():
            param.requires_grad = False

        # Print backbone model information
        print("\n" + "=" * 50)
        print(f"Loaded Backbone Model: {backbone_type}")
        print(f"Pretrained Model Name: {pretrained_model_name}")
        print(f"Backbone Hidden Size: {backbone_hidden_size}")
        print("=" * 50 + "\n")

        def initialize_weights(module):
            """
            Initialize weights for the given module using appropriate strategies:
            - Xavier initialization for linear layers
            - Kaiming initialization for layers followed by ReLU
            - Zero initialization for biases
            """
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)  # Xavier for linear layers
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)  # Zero biases
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")  # Kaiming for convolutional layers
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # ====== Build and Init Riesz and Sentiment Heads  ========
        self.riesz = RieszHead(
            backbone_hidden_size=backbone_hidden_size,
            hidden_size=backbone_hidden_size // 2,
            head_type=riesz_head_type
        )
        self.riesz.apply(initialize_weights)
        for param in self.riesz.parameters():
            param.requires_grad = True

        self.sentiment = SentimentHead(
            backbone_hidden_size=backbone_hidden_size, 
            hidden_size=backbone_hidden_size // 2,
            head_type=sentiment_head_type, 
            probs=False
        )
        self.sentiment.apply(initialize_weights)
        for param in self.sentiment.parameters():
            param.requires_grad = True
            
    def percentage_trainable_params(self):
        """         
        Returns the percentage of trainable parameters (float).
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        return trainable_percentage
    
    def percentage_trainable_component_params(self, component_name: str):
        """
        Returns the percentage of trainable backbone parameters (float).
        """
        if not component_name in ["backbone_riesz", "backbone_sentiment", "riesz", "sentiment"]:
            raise ValueError(f"Invalid component name: {component_name}. Must be one of 'backbone_riesz', 'backbone_sentiment', 'riesz', or 'sentiment'.")
        
        component = getattr(self, component_name)
        total_params = sum(p.numel() for p in component.parameters())
        trainable_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        return trainable_percentage
            
    def unfreeze_backbone_component_fraction(self, fraction: float, component_str: str):
        """ 
        Unfreeze a fraction of the backbone layers. Only considers top-level 
        layers. Rounds to an integer number of layers and unfreezes.
        
        Parameters:
        - fraction: float
            Fraction of backbone layers to unfreeze.
        - which_component: str
            Which backbone to unfreeze. Options: 'backbone_riesz', 'backbone_sentiment'
            
        Returns tuple of the percentages of trainable parameters in the backbone and overall model.
        """
        eps = 1e-2
        
        backbone = getattr(self, component_str)
        
        encoder = getattr(backbone, "encoder", getattr(backbone, "transformer", None))
        
        if encoder is None:
            raise AttributeError("The backbone model does not have an 'encoder' or 'transformer' attribute.")
            
        total_backbone_layers = len(list(encoder.layer))
        
        num_layers = None
        if fraction > 1.0 - eps:
            num_layers_to_unfreeze = 'all'
        else:
            num_layers_to_unfreeze = int(total_backbone_layers * fraction)
        
        return self.unfreeze_backbone(component = backbone, num_layers=num_layers_to_unfreeze)

    def unfreeze_backbone(self, 
                        component: torch.nn.Module,
                        num_layers: Union[str, int] = 'all', 
                        verbose: bool = False):
        """ 
        Iteratively unfreeze backbone layers.

        Parameters:
        - component: The backbone module to unfreeze.
        - num_layers: Union[str, int], default='all'
            Number of backbone layers to unfreeze.
            If 'all', unfreeze all layers. 
            
        Returns dict of the percentages of trainable parameters in the overall model and backbone.
        """
        backbone = component
        if isinstance(num_layers, int) and num_layers <= 0:  
            pass # don't unfreeze anything
        else:
            print("\n" + "=" * 50)
            print("Unfreezing Backbone Layers:")

            encoder = getattr(backbone, "encoder", getattr(backbone, "transformer", None))
            if encoder is None:
                raise AttributeError("The backbone model does not have an 'encoder' or 'transformer' attribute.")

            if num_layers == 'all':
                for param in backbone.parameters():
                    param.requires_grad = True
                print("  > All layers have been unfrozen.\n(This includes e.g. embeddings prior to the DistilBERT tranformer if using DistilBERT.)")
            else:
                # Unfreeze the last `num_layers` layers
                assert isinstance(num_layers, int), "num_layers must be 'all' or an integer."
                layer_list = list(encoder.layer)  # List of layers
                layers_to_unfreeze = layer_list[-num_layers:]   # would break with non-positive ints, but we check that above

                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"  > Last {num_layers} backbone layers have been unfrozen (transformer/encoder layers).")

            # Verbose unfreezing output
            if verbose:
                for name, param in backbone.named_parameters():
                    if param.requires_grad:
                        print(f"    - Unfrozen: {name}")

            total_params = sum(p.numel() for p in backbone.parameters())
            trainable_params_after = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            trainable_percentage = (trainable_params_after / total_params) * 100

            print(f"\nBackbone Parameters Summary:")
            print(f"  > Total Parameters: {total_params:,}")
            print(f"  > Trainable Parameters (After Unfreezing): {trainable_params_after:,}")
            print(f"  > Percentage Trainable: {trainable_percentage:.2f}%")
            print("=" * 50 + "\n")
        
        return {"trainable_model": self.percentage_trainable_params(),
                "trainable_backbone_riesz": self.percentage_trainable_component_params("backbone_riesz"), 
                "trainable_backbone_sentiment": self.percentage_trainable_component_params("backbone_sentiment")}

    def forward(self,
                input_ids_real, 
                input_ids_treated, 
                input_ids_control, 
                attention_mask_real, 
                attention_mask_treated, 
                attention_mask_control)-> Union[torch.Tensor, tuple]:
        
        if self.training:
            backbone = None
            backbone_strs = ("backbone_riesz", "backbone_sentiment")
            
            output_real = {k: None for k in backbone_strs}
            output_treated = {k: None for k in backbone_strs}
            output_control = {k: None for k in backbone_strs}
            
            embedding_real = {k: None for k in backbone_strs}
            embedding_treated = {k: None for k in backbone_strs}
            embedding_control = {k: None for k in backbone_strs}
            
            for backbone_str in ("backbone_riesz", "backbone_sentiment"):
                backbone = getattr(self, backbone_str)
                output_real[backbone_str] = backbone(input_ids_real, attention_mask=attention_mask_real)
                output_treated[backbone_str] = backbone(input_ids_treated, attention_mask=attention_mask_treated)
                output_control[backbone_str] = backbone(input_ids_control, attention_mask=attention_mask_control)
                
                # Produce single embedding for FCN or linear layers 
                # Retain sequence otherwise 
                
                if backbone_str == "backbone_riesz":
                    if self.riesz_head_type in ['fcn', 'linear']:
                        if isinstance(backbone, DistilBertModel):
                            embedding_real[backbone_str] = output_real[backbone_str].last_hidden_state[:, 0, :]  # CLS token embedding
                            embedding_treated[backbone_str] = output_treated[backbone_str].last_hidden_state[:, 0, :]
                            embedding_control[backbone_str] = output_control[backbone_str].last_hidden_state[:, 0, :]
                        elif isinstance(backbone, LlamaModel):
                            embedding_real[backbone_str] = output_real[backbone_str].last_hidden_state[:, -1, :]  # Last token embedding
                            embedding_treated[backbone_str] = output_treated[backbone_str].last_hidden_state[:, -1, :]
                            embedding_control[backbone_str] = output_control[backbone_str].last_hidden_state[:, -1, :]
                        else:
                            raise ValueError("[ERROR] Unsupported backbone model.")
                elif backbone_str == "backbone_sentiment":
                    if self.sentiment_head_type in ['fcn', 'linear']:
                        if isinstance(backbone, DistilBertModel):
                            embedding_real[backbone_str] = output_real[backbone_str].last_hidden_state[:, 0, :]  # CLS token embedding
                            embedding_treated[backbone_str] = output_treated[backbone_str].last_hidden_state[:, 0, :]
                            embedding_control[backbone_str] = output_control[backbone_str].last_hidden_state[:, 0, :]
                        elif isinstance(backbone, LlamaModel):
                            embedding_real[backbone_str] = output_real[backbone_str].last_hidden_state[:, -1, :]  # Last token embedding
                            embedding_treated[backbone_str] = output_treated[backbone_str].last_hidden_state[:, -1, :]
                            embedding_control[backbone_str] = output_control[backbone_str].last_hidden_state[:, -1, :]
                        else:
                            raise ValueError("[ERROR] Unsupported backbone model.")          

            # =========== Produce RR and Sentiment Outputs ===========
            for backbone_str in ("backbone_riesz", "backbone_sentiment"):
                if backbone_str == "backbone_riesz":
                    if self.riesz_head_type in ['fcn', 'linear']: # Pass pooled embeddings to fcn or linear
                        riesz_output_real = self.riesz(embedding_real[backbone_str])
                        riesz_output_treated = self.riesz(embedding_treated[backbone_str])
                        riesz_output_control = self.riesz(embedding_control[backbone_str])
                    elif self.riesz_head_type == 'conv':  # pass sequence of embeddings to conv
                        riesz_output_real = self.riesz(output_real[backbone_str].last_hidden_state)
                        riesz_output_treated = self.riesz(output_treated[backbone_str].last_hidden_state)
                        riesz_output_control = self.riesz(output_control[backbone_str].last_hidden_state)
                    else:
                        raise ValueError(f"[ERROR] Unsupported Riesz head type: {self.riesz_head_type}.")
                
                elif backbone_str == "backbone_sentiment":
                    if self.sentiment_head_type in ['fcn', 'linear']: # Pass pooled embeddings to fcn or linear
                        sentiment_output_real = self.sentiment(embedding_real[backbone_str])
                        sentiment_output_treated = self.sentiment(embedding_treated[backbone_str])
                        sentiment_output_control = self.sentiment(embedding_control[backbone_str])
                    elif self.sentiment_head_type == 'conv': # pass sequence of embeddings to conv
                        sentiment_output_real = self.sentiment(output_real[backbone_str].last_hidden_state)
                        sentiment_output_treated = self.sentiment(output_treated[backbone_str].last_hidden_state)
                        sentiment_output_control = self.sentiment(output_control[backbone_str].last_hidden_state)
                    else:
                        raise ValueError(f"[ERROR] Unsupported sentiment head type: {self.sentiment_head_type}.")

            return (sentiment_output_real, sentiment_output_treated, sentiment_output_control, 
                    riesz_output_real, riesz_output_treated, riesz_output_control)
        else:
            sb_output_real = self.backbone_sentiment(input_ids_real, attention_mask=attention_mask_real)
            
            embedding_real = None
            embeddings_real = None
            if self.sentiment_head_type in ['fcn', 'linear']: # Pass pooled embeddings to fcn or linear
                if isinstance(self.backbone_sentiment, DistilBertModel):
                    embedding_real = sb_output_real.last_hidden_state[:, 0, :]  # CLS token embedding
                elif isinstance(self.backbone_sentiment, LlamaModel):
                    embedding_real = sb_output_real.last_hidden_state[:, -1, :]  # Last token embedding
                else:
                    raise ValueError("[ERROR] Unsupported backbone model.")
            elif self.sentiment_head_type == 'conv': # pass sequence of embeddings to conv
                embeddings_real = sb_output_real.last_hidden_state
            else:
                raise ValueError(f"[ERROR] Unsupported sentiment head type: {self.sentiment_head_type}.")
            
            # pass either pooled or sequence embeddings depending on sentiment head type
            if embedding_real is not None:
                sentiment_output_real = self.sentiment(embedding_real)
            elif embeddings_real is not None:
                sentiment_output_real = self.sentiment(embeddings_real)
            else:
                raise ValueError("[ERROR] No valid embeddings found for sentiment head.")

            return sentiment_output_real

    # ==== components for partial freezing/ unfreezing ====
    # needed for interleaved training, riesz-only training, etc.   
    def freeze_component(self, component_name: str):
        """
        Freezes the parameters of the specified component and stores the original
        requires_grad state in self.frozen_grad_masks.

        Parameters:
        - component_name (str): The name of the component to freeze ('backbone', 'riesz', 'sentiment').
        """
        if component_name not in ["backbone_riesz", "backbone_sentiment", "riesz", "sentiment"]:
            raise ValueError(f"Invalid component name: {component_name}. Must be one of 'backbone_riesz', 'backbone_sentiment', 'riesz', or 'sentiment'.")

        component = getattr(self, component_name)
        
        # Store the original requires_grad state in a mask
        grad_mask = {name: param.requires_grad for name, param in component.named_parameters()}
        self.frozen_grad_masks[component_name] = grad_mask
        
        # Freeze the component
        for param in component.parameters():
            param.requires_grad = False

    def unfreeze_component(self, component_name: str):
        """
        Unfreezes the parameters of the specified component, restoring the original
        requires_grad state from self.frozen_grad_masks.

        Parameters:
        - component_name (str): The name of the component to unfreeze ('backbone', 'riesz', 'sentiment').
        """
        if component_name not in self.frozen_grad_masks:
            raise ValueError(f"No frozen state found for component '{component_name}'. Did you call freeze_component first?")
        
        component = getattr(self, component_name)
        grad_mask = self.frozen_grad_masks[component_name]
        
        # Restore the original requires_grad state
        for name, param in component.named_parameters():
            if name in grad_mask:
                param.requires_grad = grad_mask[name]

        # Clean up the mask to avoid unnecessary memory usage
        del self.frozen_grad_masks[component_name]