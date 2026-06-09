# Indian Legal Document Summarization Transformer

Learning project to train a small encoder-decoder Transformer from scratch for summarizing long Indian legal judgments, especially Supreme Court cases.

The goal is not to use a ready-made large language model. The goal is to understand and build the Transformer training pipeline step by step, inspired by Andrew Karpathy's "Let's build GPT" / nanoGPT style of learning: simple code, clear tensors, small experiments first, then scale carefully.

## Objective

Build a model that takes a long Indian legal judgment as plain text and generates a short abstractive summary in simple language.

The summary should capture:

- case background
- main legal issue
- important reasoning
- final decision or verdict

## Target Users

- law students
- legal researchers
- junior advocates
- non-experts who need a quick understanding of judgments

## Dataset

Primary dataset: **IN-Abs**

This dataset is suitable because it contains Indian Supreme Court case documents paired with abstractive summaries.

Expected dataset structure:

```text
IN-Abs/
  train-data/
    judgement/
    summary/
    stats-IN-train.txt
  test-data/
    judgement/
    summary/
    stats-IN-test.txt
```

For training:

```text
input  = IN-Abs/train-data/judgement/*.txt
target = IN-Abs/train-data/summary/*.txt
```

For testing:

```text
input  = IN-Abs/test-data/judgement/*.txt
target = IN-Abs/test-data/summary/*.txt
```

Each judgment file should have a matching summary file with the same filename.

## Why IN-Abs?

IN-Abs is the correct MVP dataset because:

- it is focused on Indian Supreme Court judgments
- it contains abstractive summaries, not only extracted sentences
- it has enough examples for a small supervised summarization experiment
- it directly matches the project goal: judgment text to short summary

Other datasets:

- **IN-Ext**: useful later for extractive summarization or rhetorical role analysis, but too small for the first abstractive training run.
- **UK-Abs**: useful for future cross-country experiments, but not part of the Indian legal MVP.

## Model Plan

The model will be a small Transformer trained from scratch.

Planned architecture:

- tokenizer trained on the dataset
- token embedding layer
- positional embedding layer
- Transformer encoder
- Transformer decoder
- cross-attention from decoder to encoder output
- language modeling head for summary generation

This is different from nanoGPT because nanoGPT is decoder-only, while summarization is better suited for an encoder-decoder model. However, the implementation style should stay nanoGPT-like:

- simple PyTorch code
- clear tensor shapes
- minimal abstractions
- train loop written manually
- frequent small experiments
- understandable model components

## Hardware Constraint

GPU memory: **4GB**