import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
from collections import Counter

from name_placeholder_utils import (
    insert_placeholders_for_inference,
    restore_entities,
    tokenize,
    get_entity_placeholders
)

# =====================
# VOCAB
# =====================
class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()

    def __len__(self):
        return len(self.word2idx)
    
    def get_placeholder_ids(self):
        """Get IDs of all placeholder tokens"""
        placeholders = get_entity_placeholders()
        return {self.word2idx[p] for p in placeholders if p in self.word2idx}


# =====================
# MODEL - Must match training script architecture
# =====================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        d_model = 256  # MUST MATCH TRAINING
        
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=384,
            batch_first=True
        )
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.src_emb(src)
        tgt = self.tgt_emb(tgt)
        
        tgt_mask = self.tf.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)
        
        out = self.tf(src, tgt, tgt_mask=tgt_mask)
        return self.fc(out)

# =====================
# TRANSLATION WITH GREEDY PLACEHOLDER DECODING
# =====================
def translate(model, sentence, src_vocab, tgt_vocab, device, max_len=50):
    """
    Translate with GREEDY decoding to ensure placeholder copying.
    
    Key changes:
    1. Use argmax (greedy) instead of sampling
    2. Strong penalty against reusing placeholders
    3. Boost placeholder probabilities when they appear in source
    """
    model.eval()
    
    # Step 1: Replace entities with placeholders
    preprocessed, entity_map = insert_placeholders_for_inference(sentence)
    
    print(f"  [DEBUG] Original:      {sentence}")
    print(f"  [DEBUG] Preprocessed:  {preprocessed}")
    print(f"  [DEBUG] Entity map:    {entity_map}")
    
    # Step 2: Tokenize and convert to indices
    src_tokens = tokenize(preprocessed)
    
    # Get all placeholder IDs
    placeholder_ids = tgt_vocab.get_placeholder_ids()
    
    # Find which placeholders are in the source
    src_placeholders_present = set()
    for token in src_tokens:
        if token in tgt_vocab.word2idx and tgt_vocab.word2idx[token] in placeholder_ids:
            src_placeholders_present.add(tgt_vocab.word2idx[token])
    
    print(f"  [DEBUG] Source placeholders: {[tgt_vocab.idx2word[p] for p in src_placeholders_present]}")
    
    src_ids = [src_vocab.word2idx.get(t, 3) for t in src_tokens]
    src = torch.tensor([src_ids], device=device)

    tgt_ids = [1]  # <sos>
    used_placeholders = set()
    token_counts = {}  # Track repetition

    # Step 3: GREEDY generation with placeholder tracking and repetition penalty
    with torch.no_grad():
        for step in range(max_len):
            tgt = torch.tensor([tgt_ids], device=device)
            logits = model(src, tgt)[0, -1].clone()

            # STRONG penalty for reusing placeholders
            for used_ph in used_placeholders:
                logits[used_ph] = -1e10  # Make it impossible to select again
            
            # PREVENT generating placeholders that aren't in the source
            for ph_id in placeholder_ids:
                if ph_id not in src_placeholders_present:
                    logits[ph_id] = -1e10  # Don't generate placeholders not in input
            
            # REPETITION PENALTY: Penalize tokens that appear too often
            for token_id, count in token_counts.items():
                if count >= 2:  # If token appeared 2+ times already
                    logits[token_id] -= 3.0 * count  # Increasing penalty
            
            # BOOST placeholders that should be in output but haven't been used
            remaining_placeholders = src_placeholders_present - used_placeholders
            if remaining_placeholders:
                for ph_id in remaining_placeholders:
                    logits[ph_id] += 5.0  # Boost unused placeholders
            
            # GREEDY: Take the most likely token (argmax)
            next_token = logits.argmax().item()

            if next_token == 2:  # <eos>
                break

            # Track placeholder usage
            if next_token in placeholder_ids:
                used_placeholders.add(next_token)
                print(f"  [DEBUG] Generated placeholder: {tgt_vocab.idx2word[next_token]}")
            
            # Track token repetition
            token_counts[next_token] = token_counts.get(next_token, 0) + 1

            tgt_ids.append(next_token)

    # Step 4: Reconstruct translation
    words = [tgt_vocab.idx2word.get(i, "") for i in tgt_ids[1:] if i in tgt_vocab.idx2word]
    translated = " ".join(words)
    
    print(f"  [DEBUG] Translated:    {translated}")
    
    # Step 5: Restore original entities
    final_translation = restore_entities(translated, entity_map)
    
    print(f"  [DEBUG] Final:         {final_translation}")
    print()
    
    return final_translation


# =====================
# UI HELPERS
# =====================
def wrap_text(text, font, max_width):
    words = text.split()
    lines, current = [], []
    for w in words:
        test = " ".join(current + [w])
        if font.size(test)[0] <= max_width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


# =====================
# MAIN
# =====================
def main():
    print("=" * 60)
    print("🚀 English → Spanish Translator (FIXED Entity Preservation)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model...")
    try:
        ckpt = torch.load("translator_placeholder_fixed_fast.pth", map_location=device, weights_only=False)
        src_vocab = ckpt["src_vocab"]
        tgt_vocab = ckpt["tgt_vocab"]

        model = Transformer(len(src_vocab), len(tgt_vocab)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabularies: EN={len(src_vocab)}, ES={len(tgt_vocab)}")
        
        # Verify placeholders
        en_placeholders = src_vocab.get_placeholder_ids()
        es_placeholders = tgt_vocab.get_placeholder_ids()
        
        print(f"  EN placeholders in vocab: {len(en_placeholders)}")
        print(f"  ES placeholders in vocab: {len(es_placeholders)}")
        
        if len(en_placeholders) == 0 or len(es_placeholders) == 0:
            print("\n⚠️  WARNING: Placeholders NOT found in vocabulary!")
            print("   Please retrain using the fixed training script.")
            return
        
        print(f"  ✓ Placeholders verified in vocab")
        print()
    except FileNotFoundError:
        print(f"❌ Error: 'translator_entity_fixed.pth' not found!")
        print("\nPlease train the model first using the fixed training script.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test the entity preservation system
    print("\n" + "=" * 60)
    print("Testing entity preservation with GREEDY decoding:")
    print("-" * 60)
    
    test_sentences = [
        "Hello my name is Bob",
        "Alice and Charlie are friends",
        "Maria went to the store",
        "I met David yesterday",
        "Bob and Alice went to New York"
    ]
    
    for test in test_sentences:
        print(f"\nEN: {test}")
        result = translate(model, test, src_vocab, tgt_vocab, device)
        print(f"ES: {result}")
    
    print("\n" + "=" * 60)
    print("Starting translator interface...")
    pygame.init()

    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("English → Spanish Translator (FIXED)")

    font = pygame.font.Font(None, 28)
    title_font = pygame.font.Font(None, 46)
    small_font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    input_text = ""
    translation = ""
    translating = False

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and input_text.strip() and not translating:
                    translating = True
                    translation = translate(
                        model, input_text, src_vocab, tgt_vocab, device
                    )
                    translating = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                    translation = ""
                elif event.key == pygame.K_ESCAPE:
                    input_text = ""
                    translation = ""
                else:
                    if len(input_text) < 200:
                        input_text += event.unicode

        # Draw background
        screen.fill((30, 30, 40))

        # Draw title
        title = title_font.render("Translator (FIXED)", True, (120, 255, 120))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))

        # Draw instructions
        instructions = small_font.render(
            "Type in English and press ENTER | ESC to clear | Names preserved with GREEDY decoding",
            True, (150, 255, 150)
        )
        screen.blit(instructions, (WIDTH // 2 - instructions.get_width() // 2, 70))

        # Draw English input
        screen.blit(font.render("English:", True, (180, 180, 180)), (50, 110))
        pygame.draw.rect(screen, (80, 80, 100), (50, 140, 700, 50), 2, border_radius=5)
        
        input_lines = wrap_text(input_text, font, 680)
        y_offset = 150
        for line in input_lines[:2]:
            screen.blit(font.render(line, True, (255, 255, 255)), (60, y_offset))
            y_offset += 30

        # Draw translation
        if translating:
            screen.blit(font.render("Translating...", True, (255, 200, 100)), (50, 220))
        elif translation:
            screen.blit(font.render("Spanish:", True, (180, 180, 180)), (50, 220))
            pygame.draw.rect(screen, (50, 50, 60), (50, 250, 700, 300), border_radius=5)
            pygame.draw.rect(screen, (100, 255, 100), (50, 250, 700, 300), 2, border_radius=5)

            y = 270
            for line in wrap_text(translation, font, 680):
                screen.blit(font.render(line, True, (200, 255, 200)), (60, y))
                y += 35
                if y > 520:
                    break

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()