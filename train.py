import wandb
import torch
from torch.utils.data import DataLoader
from transliteration_dataset import (
    TransliterationDataset, read_data, build_vocab, collate_fn
)
from model import Encoder, Decoder, Seq2Seq
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    # Debugging the batch structure
    print("Checking dataloader structure...")
    sample_batch = next(iter(dataloader))
    print("Length of sample batch:", len(sample_batch))
    print("Sample batch types:", type(sample_batch[0]), type(sample_batch[1]))

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == tgt).sum().item()
        total += tgt.size(0)

    acc = correct / total
    return epoch_loss / len(dataloader), acc


def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Load paths
        train_path = '/content/drive/MyDrive/Colab Notebooks/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
        dev_path = '/content/drive/MyDrive/Colab Notebooks/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'

        # Load data
        train_latins, train_targets = read_data(train_path)
        dev_latins, dev_targets = read_data(dev_path)

        # Build vocab
        src_vocab = build_vocab(train_latins + dev_latins)
        tgt_vocab = build_vocab(train_targets + dev_targets)

        # Datasets & Dataloaders
        train_dataset = TransliterationDataset(train_latins, train_targets, src_vocab, tgt_vocab)
        dev_dataset = TransliterationDataset(dev_latins, dev_targets, src_vocab, tgt_vocab)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        encoder = Encoder(
            vocab_size=len(src_vocab),
            embedding_dim=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.enc_layers,
            cell_type=config.cell_type,
        )

        decoder = Decoder(
            vocab_size=len(tgt_vocab),
            embedding_dim=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.dec_layers,
            cell_type=config.cell_type,
        )

        model = Seq2Seq(encoder, decoder).to(device)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(5):
            loss, acc = train_model(model, train_loader, optimizer, criterion, device)
            wandb.log({"loss": loss, "accuracy": acc})


# Sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'emb_dim': {'values': [32, 64, 128]},
        'hidden_dim': {'values': [64, 128]},
        'enc_layers': {'values': [1, 2]},
        'dec_layers': {'values': [1, 2]},
        'cell_type': {'values': ['RNN', 'LSTM', 'GRU']},
        'dropout': {'values': [0.2, 0.3]}
    }
}

# Launch sweep
sweep_id = wandb.sweep(sweep_config, project="Deep Learning Assignment 3")
wandb.agent(sweep_id, function=sweep_train, count=20)
