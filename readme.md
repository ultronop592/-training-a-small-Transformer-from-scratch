# Indian Legal Document Summarization Transformer

A learning project to train a small encoder-decoder Transformer from scratch
for summarizing Indian Supreme Court judgments.

Built in the style of Karpathy's nanoGPT — simple PyTorch code, clear tensor
shapes, no pretrained models, everything written and understood from scratch.

Status: Phase 1 complete (10,000 iters). Phase 2 in progress (30,000 iters).

---

## What it does

Takes a long Indian Supreme Court judgment as plain text and generates a short
abstractive summary covering:

- case background
- main legal issue
- key reasoning
- final verdict

---

## Architecture

Encoder-Decoder Transformer trained from scratch.

    Total Parameters      : 8.37M
    Embedding Dimension   : 256
    Attention Heads       : 4  (head dim = 64)
    Encoder Layers        : 3
    Decoder Layers        : 3
    Feedforward Dimension : 512
    Encoder Input Length  : 1024 tokens  (judgment)
    Decoder Output Length : 256 tokens   (summary)
    Vocabulary Size       : 8000         (BPE trained on corpus)
    Activation            : GELU
    Positional Encoding   : Learned embeddings
    Weight Tying          : lm_head shares decoder embedding weights

Three types of attention:

    Encoder self-attention   — judgment tokens attend to each other (no mask)
    Decoder self-attention   — summary tokens attend to past summary tokens (causal mask)
    Cross-attention          — summary tokens attend to full encoder output

---

## Dataset

IN-Abs — Indian Supreme Court judgments paired with abstractive summaries.

    Train pairs : 6,325
    Val pairs   : 703
    Total docs  : 7,028

Structure expected:

    Dataset/
        train-data/
            judgement/   <- input .txt files
            summary/     <- target .txt files
        test-data/
            judgement/
            summary/

---

## Training

    Optimizer      : AdamW (β1=0.9, β2=0.95)
    Learning Rate  : 3e-4 with cosine decay + linear warmup (400 steps)
    Batch Size     : 4 x 8 gradient accumulation = effective 32
    Precision      : float16 (AMP)
    Grad Clip      : 1.0
    Weight Decay   : 0.1
    Hardware       : NVIDIA RTX 2050 (4GB VRAM)

Phase 1 results (10,000 iterations):

    Train Loss     : 4.7856
    Val Loss       : 4.9168
    Training Time  : ~2.7 hours

---

## Inference

Beam search with repetition and length penalty.

    Beam Size          : 4
    Repetition Penalty : 1.3
    Length Penalty     : 0.8
    Min Length         : 30 tokens
    Temperature        : 1.0

---

## Project Structure

    Tranformer/
        config/
            train_inabs.py      hyperparameters — edit this to change anything
        data/
            inabs/
                prepare.py      tokenizer training + binary data preparation
        Dataset/
            train-data/         raw dataset (not committed to git)
            test-data/
        out-inabs/
            ckpt.pt             saved checkpoint (not committed to git)
        configurator.py         nanoGPT-style config override system
        model.py                full encoder-decoder Transformer architecture
        train.py                training loop
        sample.py               inference — judgment in, summary out
        requirements.txt

---

## How to run

    # 1. install dependencies
    pip install torch numpy tokenizers rouge-score

    # 2. prepare data — trains BPE tokenizer, writes .bin files
    python data\inabs\prepare.py

    # 3. train from scratch
    python train.py config\train_inabs.py

    # 4. resume training from checkpoint
    # set init_from = 'resume' in config\train_inabs.py, then:
    python train.py config\train_inabs.py

    # 5. generate a summary (beam search, default)
    python sample.py

    # 6. generate with custom settings
    python sample.py --beam_size 8 --rep_penalty 1.5
    python sample.py --top_k 10 --rep_penalty 1.3
    python sample.py --judgment Dataset\train-data\judgement\1.txt

---

## What is built from scratch

    BPE tokenizer trained on the legal corpus
    Token + positional embeddings
    Multi-head self-attention (encoder)
    Multi-head causal self-attention (decoder)
    Cross-attention (decoder queries, encoder keys/values)
    Layer normalization + residual connections
    FeedForward blocks (GELU activation)
    Full training loop with gradient accumulation
    AMP float16 training with GradScaler
    Cosine LR schedule with linear warmup
    Checkpoint save and resume
    Beam search with repetition and length penalty

---

## Roadmap

    Phase 1 — complete
        10,000 training iterations
        Beam search inference
        Repetition + length penalty

    Phase 2 — in progress
        Label smoothing (cross-entropy improvement)
        Resume training to 30,000 iterations
        ROUGE-1, ROUGE-2, ROUGE-L evaluation script

    Phase 3 — planned
        Scale model to d_model=512, ~30M parameters
        Coverage mechanism (reduce repetition at model level)
        RoPE positional encoding
        REINFORCE fine-tuning with ROUGE reward

---

## Why not use a pretrained model

The goal is to understand the full pipeline — tokenizer, embeddings, attention,
cross-attention, loss, gradient accumulation, checkpointing — by building each
piece from scratch, the same way Karpathy builds GPT in his video series.

Using a pretrained model would skip all of that.

---

## References

    Attention Is All You Need — Vaswani et al. 2017
    nanoGPT — Andrej Karpathy
    IN-Abs dataset — Indian Legal NLP