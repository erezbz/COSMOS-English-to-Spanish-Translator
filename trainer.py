"""
Complete English-to-Spanish Seq2Seq Training Script with Attention
Trains from scratch with proper hyperparameters
"""
from typing import List, Dict, Optional
import re
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd
import pickle
import torch.optim as optim

# =========================
# TOKENIZATION / VOCAB
# =========================
class Vocabulary:
    def __init__(self, min_freq: int = 2):  # Changed to 2 to reduce vocab size
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = None
        self.unk_id = None
        self.bos_id = None
        self.eos_id = None

    def build(self, tokenized_texts: List[List[str]]):
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens first
        for idx, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
        
        # Count token frequencies
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        
        # Apply min_freq filtering and sort
        filtered_tokens = [tok for tok, freq in counter.items() if freq >= self.min_freq]
        filtered_tokens.sort()
        
        # Assign IDs
        next_id = len(self.special_tokens)
        for tok in filtered_tokens:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = next_id
                self.id_to_token[next_id] = tok
                next_id += 1
        
        # Cache special token IDs
        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        
        assert self.pad_id == 0, "pad_id must be 0"

    def encode(self, tokens: List[str]) -> List[int]:
        if not tokens:
            return []
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        tokens = []
        for id_ in ids:
            if id_ == self.eos_id:
                break
            token = self.id_to_token.get(id_, self.unk_token)
            if token not in [self.pad_token, self.bos_token]:
                tokens.append(token)
        return tokens

def normalize_text(text: str) -> str:
    """Normalize text for consistent tokenization"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove all punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization"""
    return text.split()

# =========================
# DATASET
# =========================
class Seq2SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_vocab, tgt_vocab, max_len: int = 30):
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.src_sequences = []
        self.tgt_sequences = []

        for _, row in self.df.iterrows():
            src_tokens = tokenize(normalize_text(row['english']))
            tgt_tokens = tokenize(normalize_text(row['spanish']))
            
            # Add BOS/EOS to target
            tgt_with_special = [tgt_vocab.bos_token] + tgt_tokens + [tgt_vocab.eos_token]
            
            src_ids = src_vocab.encode(src_tokens)
            tgt_ids = tgt_vocab.encode(tgt_with_special)
            
            src_ids, src_mask = self._pad_sequence(src_ids, max_len, src_vocab.pad_id)
            tgt_ids, tgt_mask = self._pad_sequence(tgt_ids, max_len, tgt_vocab.pad_id)
            
            self.src_sequences.append((src_ids, src_mask))
            self.tgt_sequences.append((tgt_ids, tgt_mask))

    def _pad_sequence(self, ids, max_len, pad_id):
        length = len(ids)
        if length >= max_len:
            return ids[:max_len], [1]*max_len
        return ids + [pad_id]*(max_len-length), [1]*length + [0]*(max_len-length)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_ids, src_mask = self.src_sequences[idx]
        tgt_ids, tgt_mask = self.tgt_sequences[idx]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.long),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.long)
        }

def collate_fn(batch, src_vocab_pad_id, tgt_vocab_pad_id):
    batch_size = len(batch)
    max_src_len = max(item['src_ids'].size(0) for item in batch)
    max_tgt_len = max(item['tgt_ids'].size(0) for item in batch)
    
    batched_src_ids = torch.full((batch_size, max_src_len), src_vocab_pad_id, dtype=torch.long)
    batched_tgt_ids = torch.full((batch_size, max_tgt_len), tgt_vocab_pad_id, dtype=torch.long)
    batched_src_mask = torch.zeros((batch_size, max_src_len), dtype=torch.long)
    batched_tgt_mask = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)
    
    for i, item in enumerate(batch):
        src_len = item['src_ids'].size(0)
        tgt_len = item['tgt_ids'].size(0)
        batched_src_ids[i, :src_len] = item['src_ids']
        batched_src_mask[i, :src_len] = item['src_mask']
        batched_tgt_ids[i, :tgt_len] = item['tgt_ids']
        batched_tgt_mask[i, :tgt_len] = item['tgt_mask']
    
    return {
        "src_ids": batched_src_ids,
        "tgt_ids": batched_tgt_ids,
        "src_mask": batched_src_mask,
        "tgt_mask": batched_tgt_mask
    }

# =========================
# MODEL WITH ATTENTION
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hidden_size]
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Normalize with softmax
        return F.softmax(attention, dim=1)

class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=src_pad_id)
        self.encoder_rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, 
                                   dropout=dropout if num_layers > 1 else 0)
        
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=tgt_pad_id)
        self.attention = Attention(hidden_size)
        self.decoder_rnn = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers=num_layers, 
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.output_layer = nn.Linear(hidden_size * 2, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.tgt_vocab = None

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        batch_size = src_ids.size(0)
        
        # Encode
        src_embedded = self.dropout(self.encoder_embedding(src_ids))
        encoder_outputs, (hidden, cell) = self.encoder_rnn(src_embedded)
        
        # Decode
        if tgt_ids is not None:
            tgt_len = tgt_ids.size(1)
        else:
            tgt_len = 30
        
        logits = torch.zeros(batch_size, tgt_len, self.output_layer.out_features, device=src_ids.device)
        input_ids = torch.full((batch_size,), self.tgt_vocab.bos_id, dtype=torch.long, device=src_ids.device)
        
        for t in range(tgt_len):
            # Get attention weights
            attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
            
            # Apply attention to encoder outputs
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size]
            
            # Embed current input
            input_embedded = self.dropout(self.decoder_embedding(input_ids)).unsqueeze(1)
            
            # Concatenate embedded input with context
            rnn_input = torch.cat((input_embedded, context), dim=2)
            
            # Pass through decoder RNN
            decoder_output, (hidden, cell) = self.decoder_rnn(rnn_input, (hidden, cell))
            
            # Concatenate decoder output with context for prediction
            output = torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)
            output_step = self.output_layer(output)
            
            logits[:, t, :] = output_step
            
            if tgt_ids is not None and t < tgt_ids.size(1) - 1 and torch.rand(1).item() < teacher_forcing_ratio:
                input_ids = tgt_ids[:, t + 1]
            else:
                input_ids = output_step.argmax(dim=1)
        
        return logits

# =========================
# TRAINING FUNCTION
# =========================
def train_seq2seq(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: str
):
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Decrease teacher forcing over time
        teacher_forcing_ratio = max(0.5 - (epoch * 0.02), 0.1)
        
        for batch_idx, batch in enumerate(data_loader, start=1):
            src_ids = batch['src_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            optimizer.zero_grad()
            output_logits = model(src_ids, src_mask, tgt_ids, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Shift for loss calculation
            tgt_labels = tgt_ids[:, 1:].contiguous()
            output_logits = output_logits[:, :-1, :].contiguous()
            
            output_logits_flat = output_logits.view(-1, output_logits.size(-1))
            tgt_labels_flat = tgt_labels.view(-1)
            
            loss = criterion(output_logits_flat, tgt_labels_flat)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} (TF: {teacher_forcing_ratio:.2f})")
    
    return model

# =========================
# MAIN TRAINING SCRIPT
# =========================
if __name__ == "__main__":
    print("="*70)
    print("ENGLISH-TO-SPANISH SEQ2SEQ TRAINING WITH ATTENTION")
    print("="*70)
    
    # =========================
    # 1. LOAD DATA
    # =========================
    print("\n[1/6] Loading data...")
    csv_path = "englishSpanish_Dataset.csv"
    df = pd.read_csv(csv_path)
    
    print(f"  Total rows in dataset: {len(df)}")
    
    # Drop missing values
    df = df.dropna(subset=['english', 'spanish'])
    
    # Sample data
    num_samples = min(270000, len(df))
    df = df.sample(num_samples, random_state=42).reset_index(drop=True)
    print(f"  Using {len(df)} samples for training")
    
    # =========================
    # 2. BUILD VOCABULARIES
    # =========================
    print("\n[2/6] Building vocabularies...")
    
    src_tokenized = [tokenize(normalize_text(s)) for s in df['english']]
    tgt_tokenized = [tokenize(normalize_text(t)) for t in df['spanish']]
    
    src_vocab = Vocabulary(min_freq=2)  # min_freq=2 to reduce vocab size
    src_vocab.build(src_tokenized)
    
    tgt_vocab = Vocabulary(min_freq=2)
    tgt_vocab.build(tgt_tokenized)
    
    print(f"  Source vocabulary size: {len(src_vocab.token_to_id)}")
    print(f"  Target vocabulary size: {len(tgt_vocab.token_to_id)}")
    
    # =========================
    # 3. CREATE DATASET
    # =========================
    print("\n[3/6] Creating dataset...")
    
    dataset = Seq2SeqDataset(df, src_vocab, tgt_vocab, max_len=30)
    batch_size = 64  # Reduced for attention model
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_id, tgt_vocab.pad_id)
    )
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(loader)}")
    
    # =========================
    # 4. CREATE MODEL
    # =========================
    print("\n[4/6] Creating model with attention...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model = Seq2SeqModel(
        src_vocab_size=len(src_vocab.token_to_id),
        tgt_vocab_size=len(tgt_vocab.token_to_id),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        embed_size=256,
        hidden_size=512,
        num_layers=3,
        dropout=0.3  # Increased dropout
    )
    model.tgt_vocab = tgt_vocab
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # =========================
    # 5. TRAIN MODEL
    # =========================
    print("\n[5/6] Training model...")
    print("  This will take 20-35 minutes on RTX 4060")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    
    num_epochs = 25
    trained_model = train_seq2seq(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device
    )
    
    # =========================
    # 6. SAVE MODEL
    # =========================
    print("\n[6/6] Saving model and vocabularies...")
    
    torch.save(trained_model.state_dict(), "seq2seq_model.pth")
    
    with open("src_vocab.pkl", "wb") as f:
        pickle.dump({
            'token_to_id': src_vocab.token_to_id, 
            'id_to_token': src_vocab.id_to_token
        }, f)
    
    with open("tgt_vocab.pkl", "wb") as f:
        pickle.dump({
            'token_to_id': tgt_vocab.token_to_id, 
            'id_to_token': tgt_vocab.id_to_token
        }, f)
    
    print("  ✓ Model saved to: seq2seq_model.pth")
    print("  ✓ Vocabularies saved")
    
    # =========================
    # QUICK TEST
    # =========================
    print("\n" + "="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    
    model.eval()
    test_sentences = [
        "hello",
        "good morning",
        "thank you",
        "i love you",
        "how are you"
    ]
    
    for sentence in test_sentences:
        tokens = tokenize(normalize_text(sentence))
        src_ids = [src_vocab.token_to_id.get(t, src_vocab.unk_id) for t in tokens]
        src_mask = [1] * len(src_ids)
        
        # Pad to length 30
        while len(src_ids) < 30:
            src_ids.append(src_vocab.pad_id)
            src_mask.append(0)
        
        src_tensor = torch.tensor([src_ids], device=device)
        mask_tensor = torch.tensor([src_mask], device=device)
        
        with torch.no_grad():
            logits = model(src_tensor, mask_tensor)
        
        predicted_ids = [logits[0, t].argmax().item() for t in range(logits.size(1))]
        decoded = tgt_vocab.decode(predicted_ids)
        
        print(f"EN: {sentence:20} -> ES: {' '.join(decoded) if decoded else '[empty]'}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)