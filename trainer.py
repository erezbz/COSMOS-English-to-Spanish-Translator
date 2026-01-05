import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import random
from collections import Counter
from tqdm import tqdm

from name_placeholder_utils import (
    insert_placeholders_for_inference,
    tokenize,
    get_entity_placeholders,
    add_placeholders_to_vocab
)

# =====================
# LOAD DATASET
# =====================
def load_csv_dataset(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            en, es = row[0].strip(), row[1].strip()
            if 2 <= len(en.split()) <= 20:
                pairs.append((en, es))
    return pairs


# =====================
# PREPROCESS WITH PLACEHOLDERS (CRITICAL FIX)
# =====================
def preprocess_pairs(pairs):
    processed = []
    for en, es in pairs:
        en_p, _ = insert_placeholders_for_inference(en)
        es_p, _ = insert_placeholders_for_inference(es)
        processed.append((en_p.lower(), es_p.lower()))
    return processed


# =====================
# VOCAB
# =====================
class Vocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.counts = Counter()

    def add_sentence(self, s):
        for t in tokenize(s):
            self.counts[t] += 1

    def build(self, min_count=1):
        i = len(self.word2idx)
        for w, c in self.counts.items():
            if c >= min_count and w not in self.word2idx:
                self.word2idx[w] = i
                self.idx2word[i] = w
                i += 1

    def __len__(self):
        return len(self.word2idx)


# =====================
# DATASET
# =====================
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=40):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        src_ids = [self.src_vocab.word2idx.get(t, 3) for t in tokenize(src)]
        tgt_ids = [1] + [self.tgt_vocab.word2idx.get(t, 3) for t in tokenize(tgt)] + [2]
        return (
            torch.tensor(src_ids[:self.max_len]),
            torch.tensor(tgt_ids[:self.max_len])
        )


def collate(batch):
    srcs, tgts = zip(*batch)
    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts)

    def pad(x, m):
        return torch.cat([x, torch.zeros(m - len(x), dtype=torch.long)])

    return (
        torch.stack([pad(s, max_s) for s in srcs]),
        torch.stack([pad(t, max_t) for t in tgts]),
    )


# =====================
# FAST TRANSFORMER
# =====================
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        d = 256  # smaller = faster
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=0)

        self.tf = nn.Transformer(
            d_model=d,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=384,
            batch_first=True
        )
        self.fc = nn.Linear(d, tgt_vocab)

    def forward(self, src, tgt):
        src = self.src_emb(src)
        tgt = self.tgt_emb(tgt)
        mask = self.tf.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        return self.fc(self.tf(src, tgt, tgt_mask=mask))


# =====================
# TRAIN (FAST)
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    pairs = load_csv_dataset("englishSpanish_Dataset.csv")
    pairs = preprocess_pairs(pairs)
    random.shuffle(pairs)

    pairs = pairs[:20000]  # 🔥 SMALL subset for speed

    src_vocab, tgt_vocab = Vocab(), Vocab()
    add_placeholders_to_vocab(src_vocab.counts)
    add_placeholders_to_vocab(tgt_vocab.counts)

    for e, s in pairs:
        src_vocab.add_sentence(e)
        tgt_vocab.add_sentence(s)

    src_vocab.build()
    tgt_vocab.build()

    ds = TranslationDataset(pairs, src_vocab, tgt_vocab)
    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True
    )

    model = Transformer(len(src_vocab), len(tgt_vocab)).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(50):  # 🔥 quick sanity check
        model.train()
        total = 0

        for src, tgt in tqdm(dl, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(src, tgt[:, :-1])
                loss = loss_fn(
                    out.reshape(-1, out.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += loss.item()

        print(f"Epoch {epoch+1} loss: {total / len(dl):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab
    }, "translator_placeholder_fixed_fast.pth")

    print("✅ Training done (FAST sanity run)")


if __name__ == "__main__":
    main()
