"""
Quick test to verify placeholder system works before training
"""

from collections import Counter
from name_placeholder_utils import (
    extract_entities, 
    insert_placeholders_for_inference,
    restore_entities,
    add_placeholders_to_vocab,
    get_entity_placeholders,
    tokenize
)

class Vocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = Counter()
        
    def add_sentence(self, sentence):
        for word in tokenize(sentence):
            self.word_count[word] += 1
    
    def build_vocab(self, min_count=2):
        idx = 4
        for word, count in self.word_count.items():
            if count >= min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def __len__(self):
        return len(self.word2idx)

print("=" * 60)
print("TESTING PLACEHOLDER SYSTEM")
print("=" * 60)
print()

# Test 1: Entity detection improvements
print("1️⃣  Testing improved entity detection:")
print("-" * 60)
test_inputs = [
    "Hello my name is Bob",
    "Alice and Charlie are friends",
    "Bob and Alice went to New York",
    "I met David yesterday"
]

for inp in test_inputs:
    with_placeholders, entity_map = insert_placeholders_for_inference(inp)
    print(f"Input:        {inp}")
    print(f"Placeholders: {with_placeholders}")
    print(f"Entities:     {entity_map}")
    print()

print()

# Test 2: Synthetic training data examples
print("2️⃣  Testing synthetic placeholder training:")
print("-" * 60)

# Simulate what training data looks like
original_pairs = [
    ("how are you", "como estas"),
    ("i went to the store", "fui a la tienda"),
    ("this is good", "esto es bueno")
]

print("Original training pairs (no entities):")
for eng, spa in original_pairs:
    print(f"  EN: {eng}")
    print(f"  ES: {spa}")
    print()

print("Synthetic augmented pairs (with placeholders):")
import random
random.seed(42)
placeholders = get_entity_placeholders()

for eng, spa in original_pairs[:2]:  # Show 2 examples
    eng_words = eng.split()
    spa_words = spa.split()
    
    # Insert __ent0__ at position 1
    eng_words.insert(1, "__ent0__")
    spa_words.insert(1, "__ent0__")
    
    print(f"  EN: {' '.join(eng_words)}")
    print(f"  ES: {' '.join(spa_words)}")
    print()

print()

# Test 3: Vocab includes placeholders and they don't become <unk>
print("3️⃣  Testing vocab lookup:")
print("-" * 60)

vocab = Vocab()
add_placeholders_to_vocab(vocab.word_count)

# Add some words
for sent in ["how are you __ent0__", "i went to __ent1__ store"]:
    vocab.add_sentence(sent)

vocab.build_vocab(min_count=1)

# Test lookup
test_sentence = "hello __ent0__ and __ent1__"
tokens = tokenize(test_sentence)
print(f"Sentence: {test_sentence}")
print(f"Tokens:   {tokens}")
print()

for token in tokens:
    idx = vocab.word2idx.get(token, 3)
    status = "✓ FOUND" if idx != 3 else "✗ <UNK>"
    print(f"  {token:12s} -> idx {idx:3d} {status}")

print()
print("=" * 60)
print("✅ TESTS COMPLETE")
print("=" * 60)
print()
print("Key expectations:")
print("  1. 'Hello' should NOT be treated as entity")
print("  2. Placeholders __ent0__, __ent1__ should be in vocab")
print("  3. Placeholders should NOT become <unk>")
print()
print("If all checks pass, retrain with:")
print("  rm translation_data_*")
print("  python train_with_placeholders.py")