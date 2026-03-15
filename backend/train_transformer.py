import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Canonical country order for geo encoding (must match backend inference).
GEO_COUNTRY_ORDER = [
    "US", "CA", "GB", "DE", "FR", "MX", "ES", "IT", "AU", "JP", "IN", "BR", "NL", "CN",
]


def country_to_id(country: str) -> int:
    """Map country code to numeric id for model input. Unknown -> 0."""
    c = (country or "").strip().upper()
    return GEO_COUNTRY_ORDER.index(c) if c in GEO_COUNTRY_ORDER else 0


@dataclass
class EventExample:
    device_id: str
    seq_features: torch.Tensor  # (seq_len, feature_dim)
    label: int  # 0 = legit, 1 = fraud


class FraudSequenceDataset(Dataset):
    """
    Minimal sequence dataset for fraud events.

    Expected CSV schema (you can adapt this to your real data):
        device_id, timestamp, amount, channel, country, is_fraud

    For simplicity, this example:
        - groups by device_id
        - sorts by timestamp
        - builds fixed-length windows over each device's history
        - uses the last event label in the window as the target
        - uses basic numeric features: [amount, is_night, channel_id, country_id]
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 20,
        max_rows: Optional[int] = None,
    ):
        import pandas as pd

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file '{csv_path}' not found. "
                f"Place your data there or change the path."
            )

        df = pd.read_csv(csv_path)
        if max_rows is not None:
            df = df.head(max_rows)

        # Basic cleanup / type casting
        required_cols = ["device_id", "timestamp", "amount", "channel", "country", "is_fraud"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Input CSV is missing required column '{col}'. "
                    f"Columns found: {list(df.columns)}"
                )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["device_id", "timestamp"])

        # Simple categorical encoding for "channel"
        channel_map = {c: i for i, c in enumerate(df["channel"].astype(str).unique())}
        df["channel_id"] = df["channel"].astype(str).map(channel_map).astype(int)

        # Geo: encode country with canonical order for inference compatibility
        df["country_id"] = df["country"].astype(str).str.strip().str.upper().apply(
            lambda c: GEO_COUNTRY_ORDER.index(c) if c in GEO_COUNTRY_ORDER else 0
        ).astype(int)

        # Build sequences per device
        self.examples: List[EventExample] = []
        for device_id, g in df.groupby("device_id"):
            if len(g) < seq_len:
                continue

            amounts = g["amount"].astype(float).values
            # hour-of-day -> night flag
            hours = g["timestamp"].dt.hour.values
            is_night = ((hours < 6) | (hours > 22)).astype(float)
            channel_ids = g["channel_id"].astype(int).values
            country_ids = g["country_id"].astype(int).values
            labels = g["is_fraud"].astype(int).values

            # Sliding windows
            for start in range(0, len(g) - seq_len + 1):
                end = start + seq_len
                feat_amount = amounts[start:end]
                feat_night = is_night[start:end]
                feat_channel = channel_ids[start:end]
                feat_country = country_ids[start:end]
                # feature_dim = 4 (amount, is_night, channel_id, country_id)
                feats = torch.tensor(
                    list(zip(feat_amount, feat_night, feat_channel, feat_country)),
                    dtype=torch.float32,
                )
                label = int(labels[end - 1])
                self.examples.append(EventExample(device_id=str(device_id), seq_features=feats, label=label))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        # (seq_len, feature_dim), label scalar
        return ex.seq_features, torch.tensor(ex.label, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FraudTransformer(nn.Module):
    """
    Simple Transformer encoder for sequence classification.
    Input: sequence of event features
    Output: fraud probability for the sequence / final event.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        enc = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # Use representation of last timestep
        last = enc[:, -1, :]
        logits = self.cls_head(last)
        return logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_x.size(0)
        total += batch_x.size(0)

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += float(loss.item()) * batch_x.size(0)
        total += batch_x.size(0)
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch_y).sum().item())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    import argparse
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Train Transformer fraud model")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/transactions.csv",
        help="Path to CSV with transactions",
    )
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-out",
        type=str,
        default="backend/artifacts/transformer_fraud.pt",
        help="Where to save the trained model",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    dataset = FraudSequenceDataset(
        csv_path=args.csv,
        seq_len=args.seq_len,
        max_rows=args.max_rows,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Check your CSV path, seq_len, and schema."
        )

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    feature_dim = dataset[0][0].shape[-1]
    model = FraudTransformer(feature_dim=feature_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training on {device} with {len(dataset)} examples")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss={train_loss:.4f} "
            f"- val_loss={val_loss:.4f} "
            f"- val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        torch.save(best_state, args.model_out)
        print(f"Saved best model to {args.model_out} (val_loss={best_val_loss:.4f})")
    else:
        print("No model state saved (no training iterations?)")


if __name__ == "__main__":
    main()
