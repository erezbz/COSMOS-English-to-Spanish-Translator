# Neural Network–Based English -> Spanish Translator

## Project Overview

This project explores neural networks, more specifically, a LSTM with attention, and how they are used in translation. Instead of focusing on making it be able to compete with other, well known translation programs like Google Translate, I had focused on understanding how neural networks work. This project was a learning experience and in it I have learned many skills.

---


## Motivation

I chose this project because I had wanted to understand how a simple system can understand such a complex idea, language. I had previously created simpler projects like using CNNs, but I had wanted to dig deeper with LSTMs and to understand what makes them so fit for this task.

---

## Dataset

- **Source:** Tatoeba English-Spanish translation pairs
- **Size:** Approximately 270,000 sentence pairs
- **Preprocessing:**
  - Lowercasing
  - Tokenization
  - Removal of punctuation for simplicity

---


## Limitations

This model has several limitations:

- Struggles with long or complex sentences
- Limited vocabulary coverage
- Struggles with proper nouns not found in the dataset

These limitations point to both architectural and data-related challenges.

---


## How to Run

- A pretrained model is located at model/seq2seq_model.pth
Please make sure that you have Git LFS before cloning as the actual model is around 300MB

```bash
git lfs install
git clone https://github.com/erezbz/COSMOS-English-to-Spanish-Translator.git
```

Alternatively, you can also train the model instead of downloading LFS using

```bash
pip install -r requirements.txt
python trainer.py
```

After you have everything setup, please use the following commands to test it.

```bash
pip install -r requirements.txt
python translate_script.py
```
