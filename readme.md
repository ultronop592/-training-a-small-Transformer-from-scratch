# Indian Legal Document Summarization Transformer

A learning project to train a small encoder-decoder Transformer from scratch
for summarizing Indian Supreme Court judgments.

Built in the style of Karpathy's nanoGPT — simple PyTorch code, clear tensor
shapes, no pretrained models, everything written and understood from scratch.

---

## What it does

Takes a long Indian Supreme Court judgment as plain text and generates a short
abstractive summary covering:

- case background
- main legal issue
- key reasoning
- final verdict

---

## Dataset

IN-Abs — Indian Supreme Court judgments paired with abstractive summaries.

Structure expected:

    Dataset/IN-Abs/
        train-data/
            judgement/   <- input .txt files
            summary/     <- target .txt files
        test-data/
            judgement/
            summary/

---

## Model

Small encoder-decoder Transformer trained from scratch on IN-Abs.

    vocab_size   = 8000     (BPE tokenizer trained on the corpus)
    d_model      = 256
    n_heads      = 4
    d_ff         = 512
    n_enc_layers = 3
    n_dec_layers = 3
    src_max_len  = 1024     (judgment tokens)
    tgt_max_len  = 256      (summary tokens)
    total params = ~8.36M

Fits in 4GB VRAM using float16 + gradient accumulation.

---

## Project Structure

    Tranformer/
        config/
            train_inabs.py      hyperparameters
        data/
            inabs/
                prepare.py      tokenizer training + binary data prep
        Dataset/
            IN-Abs/             raw dataset (not committed to git)
        out-inabs/              checkpoints saved here
        configurator.py         nanoGPT-style config override
        model.py                encoder-decoder Transformer
        train.py                training loop
        sample.py               inference — judgment in, summary out
        requirements.txt

---

## How to run

    # 1. install dependencies
    pip install torch numpy tokenizers rouge-score

    # 2. prepare data (trains tokenizer, writes .bin files)
    python data/inabs/prepare.py

    # 3. train
    python train.py config/train_inabs.py

    # 4. generate a summary
    python sample.py

    # or point at a specific judgment file
    python sample.py --judgment Dataset/IN-Abs/test-data/judgement/1953_123.txt

---

## Why not use a pretrained model

The goal is to understand the full training pipeline — tokenizer, embeddings,
attention, cross-attention, loss, gradient accumulation, checkpointing — by
building each piece from scratch, the same way Karpathy builds GPT in his
video series. Using a pretrained model would skip all of that.

---

## Hardware

GPU: 4GB VRAM
Training uses float16 + gradient accumulation (effective batch = 32).