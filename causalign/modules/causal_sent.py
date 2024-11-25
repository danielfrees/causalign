from transformers import DistilBertModel, LlamaModel
import torch
from causalign.modules.causal_sent_heads import RieszHead, SentimentHead
from causalign.constants import SUPPORTED_BACKBONES_LIST, HF_TOKEN
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
        
        # =========== Load backbone (DistilBERT or LLaMA) =================
        if not pretrained_model_name in SUPPORTED_BACKBONES_LIST:
            warnings.warn(f"[WARNING] Unsupported/tested model name: {pretrained_model_name}. ")
            print(f"Supported models: {SUPPORTED_BACKBONES_LIST}")
            
        if "bert" in pretrained_model_name:
            self.backbone = DistilBertModel.from_pretrained(pretrained_model_name, token=HF_TOKEN)
            backbone_type = "DistilBERT"
        elif "llama" in pretrained_model_name:
            # Example: Uncomment below if LLaMA 3.1 8B is desired
            self.backbone = LlamaModel.from_pretrained(pretrained_model_name, token=HF_TOKEN)
            backbone_type = "LLaMA"
        else:
            raise ValueError(f"[ERROR] Unsupported model name: {pretrained_model_name}. "
                            f"Expected 'bert' or 'llama' in the name.")
            
        backbone_hidden_size = self.backbone.config.hidden_size
        self.backbone_hidden_size = backbone_hidden_size

        # Freeze backbone parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Print backbone model information
        print("\n" + "=" * 50)
        print(f"Loaded Backbone Model: {backbone_type}")
        print(f"Pretrained Model Name: {pretrained_model_name}")
        print(f"Backbone Hidden Size: {backbone_hidden_size}")
        print("=" * 50 + "\n")

        # Riesz and Sentiment Heads
        self.riesz = RieszHead(backbone_hidden_size = backbone_hidden_size, 
                            head_type = riesz_head_type)
        for param in self.riesz.parameters():
            param.requires_grad = True

        self.sentiment = SentimentHead(hidden_size=64, backbone_hidden_size=backbone_hidden_size, 
                                head_type = sentiment_head_type, 
                                probs=False)
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
    
    def percentage_trainable_backbone_params(self):
        """
        Returns the percentage of trainable backbone parameters (float).
        """
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        return trainable_percentage
            
    def unfreeze_backbone_fraction(self, fraction: float):
        """ 
        Unfreeze a fraction of the backbone layers. Only considers top-level 
        layers. Rounds to an integer number of layers and unfreezes.
        
        Parameters:
        - fraction: float
            Fraction of backbone layers to unfreeze.
            
        Returns tuple of the percentages of trainable parameters in the backbone and overall model.
        """
        encoder = getattr(self.backbone, "encoder", getattr(self.backbone, "transformer", None))
        if encoder is None:
            raise AttributeError("The backbone model does not have an 'encoder' or 'transformer' attribute.")
            
        total_backbone_layers = len(list(encoder.layer))
        num_layers_to_unfreeze = int(total_backbone_layers * fraction)
        
        return self.unfreeze_backbone(num_layers=num_layers_to_unfreeze)

    def unfreeze_backbone(self, 
                        num_layers: Union[str, int] = 'all', 
                        verbose: bool = False):
        """ 
        Iteratively unfreeze backbone layers.

        Parameters:
        - num_layers: Union[str, int], default='all'
            Number of backbone layers to unfreeze.
            If 'all', unfreeze all layers. 
            
        Returns dict of the percentages of trainable parameters in the overall model and backbone.
        """
        if num_layers <= 0:  
            pass # don't unfreeze anything
        else:
            print("\n" + "=" * 50)
            print("Unfreezing Backbone Layers:")

            encoder = getattr(self.backbone, "encoder", getattr(self.backbone, "transformer", None))
            if encoder is None:
                raise AttributeError("The backbone model does not have an 'encoder' or 'transformer' attribute.")

            if num_layers == 'all':
                for param in self.backbone.parameters():
                    param.requires_grad = True
                print("  > All layers have been unfrozen.")
            else:
                # Unfreeze the last `num_layers` layers
                assert isinstance(num_layers, int), "num_layers must be 'all' or an integer."
                layer_list = list(encoder.layer)  # List of layers
                layers_to_unfreeze = layer_list[-num_layers:]   # would break with non-positive ints, but we check that above

                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"  > Last {num_layers} backbone layers have been unfrozen.")

            # Verbose unfreezing output
            if verbose:
                for name, param in self.backbone.named_parameters():
                    if param.requires_grad:
                        print(f"    - Unfrozen: {name}")

            total_params = sum(p.numel() for p in self.backbone.parameters())
            trainable_params_after = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            trainable_percentage = (trainable_params_after / total_params) * 100

            print(f"\nBackbone Parameters Summary:")
            print(f"  > Total Parameters: {total_params:,}")
            print(f"  > Trainable Parameters (After Unfreezing): {trainable_params_after:,}")
            print(f"  > Percentage Trainable: {trainable_percentage:.2f}%")
            print("=" * 50 + "\n")
        
        return {"trainable_model": self.percentage_trainable_params(),
                "trainable_backbone": self.percentage_trainable_backbone_params()}

    def forward(self,
                input_ids_real, 
                input_ids_treated, 
                input_ids_control, 
                attention_mask_real, 
                attention_mask_treated, 
                attention_mask_control):
        
        if self.training:
            backbone_output_real = self.backbone(input_ids_real, attention_mask=attention_mask_real)
            backbone_output_treated = self.backbone(input_ids_treated, attention_mask=attention_mask_treated)
            backbone_output_control = self.backbone(input_ids_control, attention_mask=attention_mask_control)

            # Process outputs for DistilBERT or LLaMA
            if isinstance(self.backbone, DistilBertModel):
                embedding_real = backbone_output_real.last_hidden_state[:, 0, :]  # CLS token embedding
                embedding_treated = backbone_output_treated.last_hidden_state[:, 0, :]
                embedding_control = backbone_output_control.last_hidden_state[:, 0, :]
            elif isinstance(self.backbone, LlamaModel):
                embedding_real = backbone_output_real.last_hidden_state[:, -1, :]  # Last token embedding
                embedding_treated = backbone_output_treated.last_hidden_state[:, -1, :]
                embedding_control = backbone_output_control.last_hidden_state[:, -1, :]
            else:
                raise ValueError("[ERROR] Unsupported backbone model.")

            # Pass embeddings through Riesz and Sentiment heads
            riesz_output_real = self.riesz(embedding_real)
            riesz_output_treated = self.riesz(embedding_treated)
            riesz_output_control = self.riesz(embedding_control)

            sentiment_output_real = self.sentiment(embedding_real)
            sentiment_output_treated = self.sentiment(embedding_treated)
            sentiment_output_control = self.sentiment(embedding_control)

            return (sentiment_output_real, sentiment_output_treated, sentiment_output_control, 
                    riesz_output_real, riesz_output_treated, riesz_output_control)
        else:
            backbone_output_real = self.backbone(input_ids_real, attention_mask=attention_mask_real)
            
            if isinstance(self.backbone, DistilBertModel):
                embedding_real = backbone_output_real.last_hidden_state[:, 0, :]  # CLS token embedding
            elif isinstance(self.backbone, LlamaModel):
                embedding_real = backbone_output_real.last_hidden_state[:, -1, :]  # Last token embedding
            else:
                raise ValueError("[ERROR] Unsupported backbone model.")
            
            sentiment_output_real = self.sentiment(embedding_real)
        
            return sentiment_output_real