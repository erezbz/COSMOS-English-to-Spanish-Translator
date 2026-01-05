"""
Entity Handling System for Neural Machine Translation

This module provides functions for:
1. TRAINING: Remove entities from sentences (extract_entities)
2. INFERENCE: Replace entities with placeholders, then restore them
3. TOKENIZATION: Simple text tokenization

The model NEVER sees actual names during training - only normal text.
At inference, we swap names in/out using placeholders.
"""

import re
from typing import Tuple, Dict, List

# =====================
# TRAINING: ENTITY REMOVAL
# =====================

def extract_entities(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract and REMOVE named entities from text for training.
    
    This function identifies capitalized words (likely names) that are NOT
    at the sentence start, and removes them completely. The model will be
    trained only on the remaining text.
    
    Args:
        text: Input sentence
    
    Returns:
        (text_without_entities, entity_map)
        
    Example:
        "Hello Bob and Alice went shopping"
        -> ("Hello and went shopping", {0: "Bob", 1: "Alice"})
        
    Note: For training, we don't use the entity_map - we just want clean text.
    """
    words = text.split()
    entities = []
    cleaned_words = []
    
    i = 0
    while i < len(words):
        word = words[i]
        clean_word = word.rstrip('.,!?;:')
        
        # Check if this is a capitalized word (potential entity)
        if clean_word and clean_word[0].isupper():
            is_sentence_start = (i == 0)
            
            # If not sentence start, it's likely a name
            if not is_sentence_start:
                # Check for multi-word entity (e.g., "New York")
                entity_parts = [word]
                j = i + 1
                
                # Look ahead for more capitalized words
                while j < len(words) and words[j] and words[j][0].isupper():
                    entity_parts.append(words[j])
                    j += 1
                
                entity = " ".join(entity_parts)
                entities.append(entity)
                i = j  # Skip all entity words
                continue
        
        cleaned_words.append(word)
        i += 1
    
    # Build entity map (mostly for debugging)
    entity_map = {str(idx): entity for idx, entity in enumerate(entities)}
    cleaned_text = " ".join(cleaned_words)
    
    return cleaned_text, entity_map


# =====================
# INFERENCE: PLACEHOLDER INSERTION
# =====================

def insert_placeholders_for_inference(text: str) -> Tuple[str, Dict[str, str]]:
    """
    For inference: Replace entities with __ENTX__ placeholders.
    
    This is ONLY used at inference time to mark where names are.
    The model will translate around these placeholders.
    
    Args:
        text: User input sentence
    
    Returns:
        (text_with_placeholders, entity_map)
    
    Example:
        "Hello Bob and Alice" 
        -> ("Hello __ENT0__ and __ENT1__", 
            {"__ENT0__": "Bob", "__ENT1__": "Alice"})
    """
    words = text.split()
    result_words = []
    entity_map = {}
    entity_count = 0
    
    # Common words that are capitalized but aren't names
    # Expanded list of common non-name words
    common_words = {
        'I', 'The', 'A', 'An', 'This', 'That', 'These', 'Those',
        'Hello', 'Hi', 'Hey', 'Yes', 'No', 'Please', 'Thanks',
        'My', 'Your', 'His', 'Her', 'Their', 'Our', 'Its'
    }
    
    i = 0
    while i < len(words):
        word = words[i]
        clean_word = word.rstrip('.,!?;:')
        
        # Check if capitalized (potential entity)
        if clean_word and clean_word[0].isupper():
            # Skip common words that aren't names
            if clean_word in common_words:
                result_words.append(word)
                i += 1
                continue
            
            # Check for multi-word entity first (e.g., "New York")
            # Look ahead for consecutive capitalized words
            entity_parts = [word]
            j = i + 1
            
            while j < len(words):
                next_word = words[j]
                next_clean = next_word.rstrip('.,!?;:')
                if next_clean and next_clean[0].isupper() and next_clean not in common_words:
                    entity_parts.append(next_word)
                    j += 1
                else:
                    break
            
            # If we found a multi-word entity OR a likely single-word name
            is_sentence_start = (i == 0)
            is_likely_name = len(clean_word) > 1 and clean_word not in common_words
            
            # Treat as entity if: 
            # 1. Multi-word (e.g., "New York")
            # 2. Not sentence start
            # 3. Sentence start but clearly a name (2+ chars, not common word)
            if len(entity_parts) > 1 or not is_sentence_start or is_likely_name:
                entity = " ".join(entity_parts)
                placeholder = f"__ent{entity_count}__"
                entity_map[placeholder] = entity
                result_words.append(placeholder)
                entity_count += 1
                i = j
                continue
        
        result_words.append(word)
        i += 1
    
    return " ".join(result_words), entity_map


# =====================
# INFERENCE: ENTITY RESTORATION
# =====================

def restore_entities(text: str, entity_map: Dict[str, str]) -> str:
    """
    Replace __ENTX__ placeholders back with original entities.
    
    Args:
        text: Translated text with placeholders
        entity_map: {placeholder: original_entity} from preprocessing
    
    Returns:
        Text with entities restored
        
    Example:
        "hola __ent0__ y __ent1__", {"__ENT0__": "Bob", "__ENT1__": "Alice"}
        -> "hola Bob y Alice"
    """
    result = text
    
    # Replace both lowercase and uppercase versions
    for placeholder, entity in entity_map.items():
        result = result.replace(placeholder.lower(), entity)
        result = result.replace(placeholder, entity)
    
    return result


# =====================
# VOCABULARY MANAGEMENT
# =====================

NUM_ENTITY_PLACEHOLDERS = 10  # Support up to 10 entities per sentence

def get_entity_placeholders() -> List[str]:
    """Get list of entity placeholder tokens."""
    return [f"__ent{i}__" for i in range(NUM_ENTITY_PLACEHOLDERS)]


def add_placeholders_to_vocab(vocab_word_count: Dict[str, int]) -> None:
    """
    Force-add entity placeholders to vocabulary word count.
    Call this BEFORE building the vocab to ensure placeholders are included.
    
    Args:
        vocab_word_count: The word_count Counter from a Vocab object
    """
    for placeholder in get_entity_placeholders():
        # Set high count to ensure they're never pruned
        vocab_word_count[placeholder] = 100000
    
    print(f"   Added {NUM_ENTITY_PLACEHOLDERS} entity placeholders to vocab")


# =====================
# TOKENIZATION
# =====================

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer that splits on spaces and separates punctuation.
    
    Args:
        text: Input text
    
    Returns:
        List of tokens
        
    Example:
        "Hello, how are you?" -> ["hello", ",", "how", "are", "you", "?"]
    """
    # Add spaces around punctuation
    text = re.sub(r"([?.!,¿¡;:])", r" \1 ", text)
    
    # Split and filter empty strings
    tokens = [t.strip() for t in text.split() if t.strip()]
    
    return tokens


# =====================
# EXAMPLE USAGE & TESTS
# =====================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTITY HANDLING SYSTEM - EXAMPLES")
    print("=" * 60)
    print()
    
    # ===== TRAINING MODE =====
    print("🎓 TRAINING MODE (remove entities):")
    print("-" * 60)
    
    train_examples = [
        "Hello Bob how are you?",
        "Alice and Charlie went to New York",
        "Maria is my friend",
        "I met David yesterday",
    ]
    
    for ex in train_examples:
        cleaned, entities = extract_entities(ex)
        print(f"Original:  {ex}")
        print(f"Cleaned:   {cleaned}")
        print(f"Entities:  {entities}")
        print()
    
    print()
    
    # ===== INFERENCE MODE =====
    print("🔮 INFERENCE MODE (use placeholders):")
    print("-" * 60)
    
    # Example: Full translation pipeline
    user_input = "Hello Bob and Alice went to New York"
    print(f"1. User input:        {user_input}")
    
    # Step 1: Insert placeholders
    with_placeholders, entity_map = insert_placeholders_for_inference(user_input)
    print(f"2. With placeholders: {with_placeholders}")
    print(f"   Entity map:        {entity_map}")
    
    # Step 2: Model translates (simulated)
    fake_translation = "hola __ent0__ y __ent1__ fueron a __ent2__"
    print(f"3. Model output:      {fake_translation}")
    
    # Step 3: Restore entities
    final = restore_entities(fake_translation, entity_map)
    print(f"4. Final result:      {final}")
    print()
    
    print()
    
    # ===== TOKENIZATION =====
    print("✂️  TOKENIZATION:")
    print("-" * 60)
    
    test_texts = [
        "Hello, how are you?",
        "I went to the store.",
        "¿Cómo estás?"
    ]
    
    for txt in test_texts:
        tokens = tokenize(txt)
        print(f"Text:   {txt}")
        print(f"Tokens: {tokens}")
        print()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)