from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datasets.generators import (
    TextAlignDataset,
    load_data
)

from causalign.modules.bert_pretrained import BertPreTrained
from causalign.modules.utils import save_model, get_training_args, seed_everything

TQDM_DISABLE = False
BERT_HIDDEN_SIZE = 768

def main(args):
    """
    Main function to run the training loop.
    """
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    nli_data = load_data(args.nli_filename)
    print("ACL Abstract Data loaded")
    dataset = TextAlignDataset(nli_data, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn,
                            num_workers=args.num_workers)
    print("ACL Abstract Dataloader created")

    # Setup config to be saved later
    config = {'tau': args.tau,
              'hidden_size': BERT_HIDDEN_SIZE,
              'data_dir': '.'}
    config = SimpleNamespace(**config)

    model = BertPreTrained().to(device)
    print("Model initialized")
    criterion = ContrastiveLearningLoss(tau=args.tau)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda')


    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader):
            token_ids_1 = batch['token_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            token_ids_2 = batch['token_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            token_ids_3 = batch['token_ids_3'].to(device)
            attention_mask_3 = batch['attention_mask_3'].to(device)

            optimizer.zero_grad()

            with autocast():
                premise_emb = model(token_ids_1, attention_mask_1)
                entailment_emb = model(token_ids_2, attention_mask_2)
                contradiction_emb = model(token_ids_3, attention_mask_3)

                loss = criterion(premise_emb, entailment_emb, contradiction_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    # Save the trained model
    save_model(model, optimizer, args, config, args.output_model_path)
    print(f"Model saved to {args.output_model_path}")

class ContrastiveLearningLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastiveLearningLoss, self).__init__()
        self.tau = tau
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, premise_emb, entailment_emb, contradiction_emb):
        """
        premise_emb: tensor of size (batch_size, embedding_size)
        entailment_emb: tensor of size (batch_size, embedding_size)
        contradiction_emb: tensor of size (batch_size, embedding_size)
        """

        cos_sim_pos = self.cosine_similarity(premise_emb, entailment_emb) # ()
        cos_sim_neg = self.cosine_similarity(premise_emb, contradiction_emb)

        scaled_logit_pos = torch.exp(torch.div(cos_sim_pos, self.tau))
        scaled_logit_neg = torch.exp(torch.div(cos_sim_neg, self.tau))

        numerator = scaled_logit_pos
        denominator = scaled_logit_pos.sum(dim=0) + scaled_logit_neg.sum(dim=0)

        output = torch.mul(torch.log(torch.div(numerator, denominator)), -1)
        return output.mean()


if __name__ == '__main__':
    args = get_training_args()
    args.output_model_path = f'params-{args.epochs}-{args.lr}-{args.tau}-contrastive-baseline.pt'  # Save path.
    seed_everything(args.seed)
    main(args)