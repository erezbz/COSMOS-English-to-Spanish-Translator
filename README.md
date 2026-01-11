# Neural Network–Based English -> Spanish Translator

## Project Overview

This projects explores neural networks, more specificly, transformers, and how they are used in translation. Instead on focusing on making it be able to compete with other, well known translation programs like google translate, I had focused on understanding how transformers work. This project was a learning experience and in it I have learned many skills.

---


## Motivation

I had chose this project because I had wanted to understand how a simple system can understad such a complex idea, language. I had previously created simpler projects like CNNs, but I had wanted to dig deeper with transformers and to understand what makes them so fit for this task.

---

## Dataset

- **Source:** tatobea english-spanish translation pairs
- **Size:** Approximately 270,000 sentence pairs
- **Preprocessing:**
  - Lowercasing
  - Tokenization
  - removal of panctuation for simplicity

---


## Limitations

This model has several limitations:

- Struggles with long or complex sentences
- Limited vocabulary coverage
- struggle with many names not found in dataset

These limitations point to both architectural and data-related challenges.

---


## How to Run

```bash
pip install -r requirements.txt
python translate_script.py
```
