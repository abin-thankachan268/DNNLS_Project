# DNNLS StoryReasoning GAN (CLIP + GPT-2)

This project predicts the next-frame caption from K context frames in the StoryReasoning dataset. The core system fuses CLIP image and text embeddings for context, generates captions with a GPT-2 prefix decoder, and optionally adds a discriminator in CLIP space for adversarial alignment.

## Task definition

Given a story of sequential frames (images and captions), predict the next-frame caption using the first K context frames. We fix K = 4 (predict frame 5) since all stories have at least 5 frames and this matches the assignment setup.

## Dataset summary

| Split | Stories |
| --- | --- |
| Train | 3552 |
| Test | 626 |

| Property | Value |
| --- | --- |
| Columns | story_id, images, frame_count, chain_of_thought, story |
| Duplicates | 0 |
| Frame count validity | frame_count equals number of images for all rows |
| Frames per story | mean 12.44, median 13, range 5 to 22 |
| Caption length | average about 80 words per frame, high variance |
| Image formatting | height 240px, width varies, wide aspect ratio |

## Data preparation

| Step | Details |
| --- | --- |
| Filtering | Keep stories with frame_count at least K+1 |
| Splitting | Story-level split with frame-count bin stratification |
| Loading | On-the-fly image loading with tqdm progress |

## Model architecture

### Shared encoder (multimodal context fusion)

For each of the K context frames:

1. Extract CLIP image embedding (512-d).
2. Extract CLIP text embedding of the context caption (512-d, truncated for stability).
3. Concatenate and project into a fused representation.
4. Run a small Transformer encoder over the K fused tokens.
5. Mean-pool to produce a single context vector z.

CLIP is frozen to keep training stable and fit within limited compute.

### Text decoder (caption generator)

GPT-2 is used with a learned prefix:

- Map context vector z to n_prefix embeddings.
- Prepend the prefix to GPT-2 token embeddings.
- GPT-2 starts frozen for prefix-only training.

### Image embedding head (auxiliary supervision)

An MLP maps z to a predicted image embedding (512-d). The loss is MSE to the CLIP embedding of the target image.

## Training objectives

### Baseline (multitask supervised)

| Loss | Description |
| --- | --- |
| L_txt | Causal LM loss on target caption tokens, prompt masked |
| L_img | MSE between predicted image embedding and target image CLIP embedding |
| L_align | 1 minus cosine similarity between predicted image embedding and CLIP text embedding of target caption |

Total loss:

```python
L_total = L_txt + alpha * L_img + lambda * L_align
```

### Adversarial variants

The discriminator operates in CLIP embedding space on (CLIP target image, CLIP text) pairs using hinge GAN losses. Three variants were explored:

| Variant | Summary | Outcome |
| --- | --- | --- |
| Naive GAN | Generate captions inside the training loop | Too slow, unstable |
| Cached-caption GAN | Generate subset once per epoch | Faster, but discriminator can exploit shortcuts |
| Hard-negative GAN | Add shuffled GT captions and cached generations, with feature matching | More stable, still marginal metric change |

## Results

### CLIPScore (generated captions)

| Model | Val | Test |
| --- | --- | --- |
| GAN_improved (cached-caption) | 0.2442 | 0.2419 |
| GAN_hardneg (hard-negative) | 0.2443 | 0.2434 |

Baseline performance is stable in the 0.24 to 0.25 CLIPScore range.

## Final conclusion

1. The multitask baseline (LM with CLIP-based auxiliary losses) is stable and effective under limited compute.
2. Adversarial variants add complexity without reliable metric gains.
3. Best tradeoff for this assignment is the baseline with strong preprocessing and robust token masking.

## Key implementation points

- RAM-safe on-the-fly dataloaders.
- Deterministic split with length-based stratification.
- Robust token masking to avoid zero supervised tokens and NaN loss.
- tqdm progress tracking in data and training loops.
- Evaluation uses CLIPScore between target images and generated captions.

## Repo layout

- `src/dataloader.py` dataset parsing, splits, and PyTorch loaders
- `src/encoders_image.py` CLIP image embedding utilities
- `src/encoders_text.py` prompt formatting and tokenization
- `src/model_dual_decoders.py` generator and discriminator modules
- `src/train.py` warmup and adversarial training loops
- `src/eval_alignment.py` CLIPScore and discriminator accuracy
- `src/download_dataset.py` dataset fetch into local cache
- `notebooks/notebook.ipynb` full experiment notebook
- `data/` local caches, runs, and outputs

## Setup

Create and activate a venv, then install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install datasets transformers accelerate evaluate tqdm pandas numpy pillow matplotlib scikit-learn wordcloud
```

## Download dataset

```bash
python -m src.download_dataset
```

## Train (warmup)

```bash
python -m src.train --mode warmup --k 4 --batch 2 --warmup_epochs 2
```

## Train (adversarial)

```bash
python -m src.train --mode adv --k 4 --batch 2 --adv_epochs 2
```

## Evaluate

```bash
python -m src.eval_alignment --checkpoint data\runs\<run_id>\GAN_final.pt
```

## Notes

- All caches and outputs are stored under `data/`.
- Use `--unfreeze_gpt` in `src/train.py` if you want to fine-tune GPT-2
