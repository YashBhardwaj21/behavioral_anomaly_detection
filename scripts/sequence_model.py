"""
Sequence-Aware Anomaly Detection via LSTM Autoencoder (PyTorch)

"sequence-aware approach (LSTM/GRU, Transformer, or graph-based) to flag deviations."

Architecture:
    Embedding(vocab_size, 16) -> LSTM(32) encoder -> LSTM(32) decoder -> Dense(vocab_size)
    Trained ONLY on 'normal' command sequences from the training time window
    (same semi-supervised philosophy as Isolation Forest).

Output:
    Per-event reconstruction error saved to models/sequence_scores.parquet,
    keyed by event_id. This score becomes an additional feature column
    consumed by LightGBM alongside the 15 causal features and the
    Isolation Forest risk score (17 total input columns).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Configuration ----
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'synthetic_access_logs.parquet')

SEQ_MAX_LEN = 10         # pad/truncate all command sequences to this length
EMBED_DIM = 16           # embedding dimension
HIDDEN_DIM = 32          # LSTM hidden size
BATCH_SIZE = 256
EPOCHS = 8               # for the hackathon only
LR = 1e-3
TEST_FRACTION = 0.20     
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---- Vocabulary ----

def build_vocab(sequences):
    """Build token -> index mapping from training sequences."""
    tokens = set()
    for seq in sequences:
        if isinstance(seq, str):
            tokens.update(seq.split("|"))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, tok in enumerate(sorted(tokens), start=2):
        vocab[tok] = i
    return vocab


def encode_sequence(seq_str, vocab, max_len):
    """Convert pipe-delimited string to fixed-length integer array."""
    if not isinstance(seq_str, str) or seq_str.strip() == "":
        return [0] * max_len
    tokens = seq_str.split("|")[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    ids += [0] * (max_len - len(ids))         # 0 = <PAD>
    return ids


# ---- Dataset ----

class SeqDataset(Dataset):
    def __init__(self, encoded_seqs):
        self.data = torch.LongTensor(encoded_seqs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # input = target (autoencoder)


# ---- Model ----

class LSTMAutoencoder(nn.Module):
    """Encoder-decoder LSTM autoencoder for command sequence reconstruction."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Encoder
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Encode
        embedded = self.embedding(x)                      # (B, L, E)
        _, (h, c) = self.encoder(embedded)                # compress to latent

        # Decode — feed the same input sequence through decoder initialized
        # with the encoder's final hidden state
        dec_out, _ = self.decoder(embedded, (h, c))       # (B, L, H)
        logits = self.fc_out(dec_out)                     # (B, L, vocab_size)
        return logits


def compute_reconstruction_error(model, encoded_seqs, batch_size=512):
    """Per-sequence mean cross-entropy reconstruction error."""
    model.eval()
    dataset = SeqDataset(encoded_seqs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    errors = []
    with torch.no_grad():
        for x, target in loader:
            logits = model(x)                             # (B, L, V)
            # Reshape for cross-entropy: (B*L, V) vs (B*L)
            loss_per_token = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            ).reshape(x.size(0), -1)                      # (B, L)

            # Mean over non-padding positions
            mask = (target != 0).float()                  # (B, L)
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            mean_error = (loss_per_token * mask).sum(dim=1) / seq_lengths
            errors.append(mean_error.numpy())

    return np.concatenate(errors)


# ---- Main Pipeline ----

def train_sequence_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load raw data
    print("--- [1/5] Loading raw access logs ---")
    df = pd.read_parquet(RAW_DATA_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} events\n")

    # 2. Time-based split
    print("--- [2/5] Time-based split ---")
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    df_train = df.iloc[:split_idx]
    print(f"Train: {len(df_train):,} | Test: {len(df) - len(df_train):,}\n")

    # 3. Build vocabulary from NORMAL training sequences only
    print("--- [3/5] Building vocabulary from normal training sequences ---")
    normal_train = df_train[df_train["label"] == "normal"]
    vocab = build_vocab(normal_train["command_sequence"])
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} tokens\n")

    with open(os.path.join(MODEL_DIR, "sequence_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # 4. Train LSTM autoencoder on normal sequences
    print("--- [4/5] Training LSTM autoencoder ---")
    normal_encoded = [encode_sequence(s, vocab, SEQ_MAX_LEN) for s in normal_train["command_sequence"]]
    train_dataset = SeqDataset(normal_encoded)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder(vocab_size, EMBED_DIM, HIDDEN_DIM, SEQ_MAX_LEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, target in train_loader:
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS} — avg loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_autoencoder.pt"))
    # Save config for inference reproducibility
    model_config = {"vocab_size": vocab_size, "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM, "max_len": SEQ_MAX_LEN}
    with open(os.path.join(MODEL_DIR, "lstm_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # 5. Score ALL events (train + test) with reconstruction error
    print("\n--- [5/5] Scoring all events with reconstruction error ---")
    all_encoded = [encode_sequence(s, vocab, SEQ_MAX_LEN) for s in df["command_sequence"]]
    errors = compute_reconstruction_error(model, all_encoded)

    scores_df = pd.DataFrame({
        "event_id": df["event_id"].values,
        "sequence_anomaly_score": errors,
    })
    output_path = os.path.join(MODEL_DIR, "sequence_scores.parquet")
    scores_df.to_parquet(output_path, index=False)

    print(f"\nSaved sequence scores to: {output_path}")
    print(f"Score stats — mean: {errors.mean():.4f}, std: {errors.std():.4f}, "
          f"max: {errors.max():.4f}")

    # Quick sanity: mean score by label
    scores_df["label"] = df["label"].values
    print("\nMean sequence_anomaly_score by label:")
    print(scores_df.groupby("label")["sequence_anomaly_score"].mean().round(4).to_string())


if __name__ == "__main__":
    train_sequence_model()
