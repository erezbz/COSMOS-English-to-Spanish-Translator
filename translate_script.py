import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import unicodedata
from typing import List, Optional

# =========================
# Vocabulary class
# =========================
class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = None
        self.unk_id = None
        self.bos_id = None
        self.eos_id = None

    def decode(self, ids: List[int]) -> List[str]:
        tokens = []
        for id_ in ids:
            if id_ == self.eos_id:
                break
            token = self.id_to_token.get(id_, self.unk_token)
            if token not in [self.pad_token, self.bos_token]:
                tokens.append(token)
        return tokens

# =========================
# Attention mechanism
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)

# =========================
# Seq2Seq model with attention
# =========================
class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_id, tgt_pad_id, 
                 embed_size=256, hidden_size=512, num_layers=2, dropout=0.3):
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

    def forward(self, src_ids, src_mask, tgt_ids=None, teacher_forcing_ratio=0.0):
        batch_size = src_ids.size(0)
        
        src_embedded = self.dropout(self.encoder_embedding(src_ids))
        encoder_outputs, (hidden, cell) = self.encoder_rnn(src_embedded)
        
        if tgt_ids is not None:
            tgt_len = tgt_ids.size(1)
        else:
            tgt_len = 30
        
        logits = torch.zeros(batch_size, tgt_len, self.output_layer.out_features, device=src_ids.device)
        input_ids = torch.full((batch_size,), self.tgt_vocab.bos_id, dtype=torch.long, device=src_ids.device)
        
        for t in range(tgt_len):
            attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            input_embedded = self.dropout(self.decoder_embedding(input_ids)).unsqueeze(1)
            rnn_input = torch.cat((input_embedded, context), dim=2)
            decoder_output, (hidden, cell) = self.decoder_rnn(rnn_input, (hidden, cell))
            output = torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)
            output_step = self.output_layer(output)
            
            logits[:, t, :] = output_step
            input_ids = output_step.argmax(dim=1)
            
            if (input_ids == self.tgt_vocab.eos_id).all():
                break
        
        return logits

# =========================
# Utilities
# =========================
def normalize_text(text: str) -> str:
    """Must match training normalization exactly"""
    if not text:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization"""
    return text.split()

def extract_entities(text: str) -> tuple[str, dict]:
    """
    Extract entity placeholders like __ent1__, __ent2__, etc.
    Returns: (text_with_placeholders, entity_map)
    """
    import re
    entity_pattern = r'__ent\d+__'
    entities = {}
    
    for match in re.finditer(entity_pattern, text):
        entity = match.group()
        entities[entity] = entity  # Store original
    
    return text, entities

def restore_entities(tokens: List[str], entity_map: dict) -> List[str]:
    """
    Replace entity placeholders in output with original entities
    """
    result = []
    for token in tokens:
        if token in entity_map:
            result.append(entity_map[token])
        else:
            result.append(token)
    return result

def reconstruct_vocab(dicts) -> Vocabulary:
    vocab = Vocabulary(min_freq=1)
    vocab.token_to_id = dicts['token_to_id']
    vocab.id_to_token = {int(k): v for k, v in dicts['id_to_token'].items()}
    vocab.pad_id = vocab.token_to_id.get("<pad>", 0)
    vocab.unk_id = vocab.token_to_id.get("<unk>", 1)
    vocab.bos_id = vocab.token_to_id.get("<bos>", 2)
    vocab.eos_id = vocab.token_to_id.get("<eos>", 3)
    return vocab

# =========================
# Load vocab and model
# =========================
print("="*70)
print("LOADING MODEL AND VOCABULARIES")
print("="*70)

print("\nLoading vocabularies...")
with open("vocab/src_vocab.pkl", "rb") as f:
    src_vocab = reconstruct_vocab(pickle.load(f))
with open("vocab/tgt_vocab.pkl", "rb") as f:
    tgt_vocab = reconstruct_vocab(pickle.load(f))

print(f"✓ Source vocab size: {len(src_vocab.token_to_id)}")
print(f"✓ Target vocab size: {len(tgt_vocab.token_to_id)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

print("\nLoading model with attention...")

# Load state dict to detect number of layers
state_dict = torch.load("model/seq2seq_model.pth", map_location=device, weights_only=False)

# Detect number of LSTM layers from state dict keys
num_layers = 1
for key in state_dict.keys():
    if 'encoder_rnn.weight_ih_l' in key:
        layer_num = int(key.split('_l')[-1]) + 1
        num_layers = max(num_layers, layer_num)

print(f"  Detected {num_layers} LSTM layers in saved model")

model = Seq2SeqModel(
    src_vocab_size=len(src_vocab.token_to_id),
    tgt_vocab_size=len(tgt_vocab.token_to_id),
    src_pad_id=src_vocab.pad_id,
    tgt_pad_id=tgt_vocab.pad_id,
    embed_size=256,
    hidden_size=512,
    num_layers=num_layers,  # Use detected number
    dropout=0.3
)
model.tgt_vocab = tgt_vocab
model.load_state_dict(state_dict)
model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded successfully ({total_params:,} parameters)")

# =========================
# Beam Search Decoding
# =========================
def beam_search_decode(model, src_ids, src_mask, beam_width=5, max_len=30, length_penalty=0.6):
    """
    Beam search decoding for better translation quality
    """
    model.eval()
    device = src_ids.device
    
    # Encode source
    src_embedded = model.dropout(model.encoder_embedding(src_ids))
    encoder_outputs, (hidden, cell) = model.encoder_rnn(src_embedded)
    
    # Initialize beams: (score, sequence, hidden, cell)
    beams = [(0.0, [model.tgt_vocab.bos_id], hidden, cell)]
    completed = []
    
    for step in range(max_len):
        candidates = []
        
        for score, seq, h, c in beams:
            # Stop if EOS already generated
            if seq[-1] == model.tgt_vocab.eos_id:
                completed.append((score, seq))
                continue
            
            # Get last token
            input_id = torch.tensor([[seq[-1]]], device=device)
            
            # Attention
            attn_weights = model.attention(h[-1], encoder_outputs, src_mask)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            
            # Decoder step
            input_embedded = model.dropout(model.decoder_embedding(input_id))
            rnn_input = torch.cat((input_embedded, context), dim=2)
            decoder_output, (new_h, new_c) = model.decoder_rnn(rnn_input, (h, c))
            
            # Output probabilities
            output = torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)
            logits = model.output_layer(output)
            log_probs = F.log_softmax(logits, dim=1)
            
            # Get top k tokens
            top_log_probs, top_indices = torch.topk(log_probs[0], beam_width)
            
            # Create new candidates
            for log_prob, idx in zip(top_log_probs, top_indices):
                new_score = score + log_prob.item()
                new_seq = seq + [idx.item()]
                
                # Penalize repetition
                if len(new_seq) >= 2 and new_seq[-1] == new_seq[-2]:
                    new_score -= 2.0  # Penalty for repeating same word
                
                candidates.append((new_score, new_seq, new_h, new_c))
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        beams = candidates[:beam_width]
        
        # Early stopping if all beams completed
        if not beams:
            break
    
    # Add remaining beams to completed
    completed.extend(beams)
    
    # Sort by score with length normalization
    completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
    
    # Return best sequence
    if completed:
        best_seq = completed[0][1]
        return best_seq[1:]  # Remove BOS token
    else:
        return []

# =========================
# Translate function
# =========================
def translate(sentence: str, use_beam_search: bool = True, verbose: bool = False):
    model.eval()
    
    # Extract entities before normalization
    text_with_entities, entity_map = extract_entities(sentence)
    
    normalized = normalize_text(text_with_entities)
    tokens = tokenize(normalized)
    
    if verbose:
        print(f"\n  Original: '{sentence}'")
        print(f"  Entities found: {list(entity_map.keys())}")
        print(f"  Normalized: '{normalized}'")
        print(f"  Tokens: {tokens}")
    
    if not tokens:
        return "[Empty input]"
    
    # Encode
    src_ids = [src_vocab.token_to_id.get(t, src_vocab.unk_id) for t in tokens]
    src_mask = [1] * len(src_ids)
    
    # Pad to length 30
    while len(src_ids) < 30:
        src_ids.append(src_vocab.pad_id)
        src_mask.append(0)
    
    src_tensor = torch.tensor([src_ids], device=device)
    mask_tensor = torch.tensor([src_mask], device=device)
    
    # Translate
    with torch.no_grad():
        if use_beam_search:
            predicted_ids = beam_search_decode(model, src_tensor, mask_tensor, beam_width=5)
        else:
            # Greedy decoding (old method)
            logits = model(src_tensor, mask_tensor)
            predicted_ids = [logits[0, t].argmax().item() for t in range(logits.size(1))]
    
    decoded = tgt_vocab.decode(predicted_ids)
    
    # Restore entities in output
    if entity_map:
        decoded = restore_entities(decoded, entity_map)
    
    if not decoded:
        return "[No translation generated]"
    
    return " ".join(decoded)

# =========================
# Interactive REPL
# =========================
def interactive_mode():
    print("\n" + "="*70)
    print("ENGLISH TO SPANISH TRANSLATION")
    print("="*70)
    print("\nCommands:")
    print("  - Type a sentence to translate")
    print("  - Type 'beam on/off' to toggle beam search (default: on)")
    print("  - Type 'verbose on/off' to toggle detailed output")
    print("  - Type 'quit' to exit")
    print()
    
    verbose = False
    use_beam = True
    
    while True:
        sentence = input("EN> ").strip()
        
        if not sentence:
            continue
        
        if sentence.lower() == "quit":
            print("\nGoodbye!")
            break
        
        if sentence.lower().startswith("beam"):
            if "on" in sentence.lower():
                use_beam = True
                print("  Beam search: ON (better quality, slower)\n")
            elif "off" in sentence.lower():
                use_beam = False
                print("  Beam search: OFF (faster, greedy decoding)\n")
            continue
        
        if sentence.lower().startswith("verbose"):
            if "on" in sentence.lower():
                verbose = True
                print("  Verbose mode: ON\n")
            elif "off" in sentence.lower():
                verbose = False
                print("  Verbose mode: OFF\n")
            continue
        
        translation = translate(sentence, use_beam_search=use_beam, verbose=verbose)
        print(f"ES> {translation}")
        print()

# =========================
# Main
# =========================
if __name__ == "__main__":
    interactive_mode()