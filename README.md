# MusicGen-small Erhu LoRA Adaptation

Supporting material for the MSc project *Domain Adaptation of MusicGen for
Erhu Timbre Adaptation via LoRA Fine-tuning* (You Wang, 250090532, MSc
Artificial Intelligence, Queen Mary University of London).

---

## 1. What this project does

The project asks a single question: **can Low-Rank Adaptation (LoRA) shift a
pre-trained text-to-music model towards the timbre of the erhu, using the
small corpus that is realistically available for a traditional instrument?**

Large text-to-music models such as MusicGen are trained overwhelmingly on
Western music, so they hold general musical knowledge without a detailed
representation of instruments that are sparsely represented in their training
data. Retraining such a model on erhu audio is not feasible — the corpus here
is 21 recordings totalling about 22.7 minutes. LoRA offers an alternative:
freeze the backbone and train a small number of additional parameters
(0.13–0.53% of the model, depending on rank).

The experiment compares a frozen MusicGen-small baseline against LoRA adapters
at ranks 4, 8 and 16, under matched prompts, generation settings and random
seeds, and scores them with Fréchet Audio Distance (FAD).

**The headline result is negative, and deliberately reported as such.** The
evaluation is two-stage: rank is selected on a validation reference set, and
the selected adapter is then scored once against a test reference set withheld
from that selection. On the validation references the rank-8 adapter scored
9.14% below the baseline; on the held-out test references it scored 0.34%
*above* it. With three validation crops, six test crops, and one training run
per rank, the design cannot distinguish a modest true effect from run-to-run
variation — so no improvement is claimed. The contribution is methodological:
a leakage-aware, seed-matched pipeline in which an apparent gain measured
during model selection is re-tested on independent references rather than
reported as a result.

Two design choices follow from this framing and are worth stating up front:

- **Timbre, not technique.** CCOM-HuQin ships playing-technique annotations,
  but they are not used anywhere in this project. A 30-second segment usually
  contains several techniques, and an annotated technique may occupy only a
  fraction of it, so pairing a segment with a prompt naming one technique
  would be weak and partly incorrect supervision. The scope is restricted to
  whether the output moves towards erhu recordings *as a class*.
- **A single fixed prompt.** All audio is paired with one description:
  *"A clean acoustic solo played by erhu, a traditional Chinese two-stringed
  bowed instrument."* Because the text condition never varies, this project
  makes no claim about prompt controllability.

---

## 2. Dataset and attribution

The audio comes from the **CCOM-HuQin** dataset:

> Zhang, Y., Zhou, Z., Li, X., Yu, F. and Sun, M. (2023) 'CCOM-HuQin: an
> annotated multimodal Chinese fiddle performance dataset', *Transactions of
> the International Society for Music Information Retrieval*, 6(1),
> pp. 60–74. doi: [10.5334/tismir.146](https://doi.org/10.5334/tismir.146)

CCOM-HuQin is a multimodal dataset of Chinese bowed-string (huqin)
performances released by the Central Conservatory of Music, with musical and
playing-technique annotations. This project uses **only the erhu subset**: 21
usable audio files, 1360.1 seconds in total.

**The dataset is not redistributed in this repository.** Obtain it from the
original source cited above, then place the erhu recordings where the notebook
expects them (see §5). Please cite the dataset authors in any work that builds
on this material, and check the licence terms attached to the original release
before redistributing any part of it.

The exact 21 files used, their grouping into 17 canonical pieces, and the
13/1/3 train/validation/test split are recorded in the manifests under
`erhu_lora_runs/<run_id>/`, so the partition can be reconstructed exactly
without guesswork.

---

## 3. Repository structure

```
250090532_YouWang_SupportingMaterial/
├── README.md                     This file
├── .gitignore
├── MSc814_YouWang.ipynb          Full pipeline, run top to bottom
└── erhu_lora_runs/               One timestamped directory per execution
    └── <run_id>/
        ├── split_manifest.json   Piece → partition assignment
        ├── segment_manifest.csv  Every 30 s segment, with source file
        ├── adapters/
        │   ├── rank_4/           LoRA weights (best-validation checkpoint)
        │   ├── rank_8/
        │   └── rank_16/
        ├── training_history/     Per-epoch train and validation loss
        ├── generated/            20 samples per condition
        └── fad_results/          Validation and test FAD tables
```

Not tracked in Git (excluded via `.gitignore`): the CCOM-HuQin archive,
extracted source audio, and generated `.wav` files. LoRA adapter weights are
small (3–13 MB) and *are* tracked, so the generation stage can be reproduced
without retraining.

---

## 4. Environment

Developed and run in **Google Colab** on a single **NVIDIA T4 GPU**. The
notebook installs its own dependencies in the first cell; no local setup is
required.

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11 |
| Transformers | 4.46.3 |
| PEFT | 0.13.2 |
| frechet-audio-distance | 0.3.4 |
| Base checkpoint | `facebook/musicgen-small` (588,981,826 params) |

```bash
pip install transformers==4.46.3 peft==0.13.2 "accelerate>=0.34,<2" \
    soundfile sentencepiece frechet-audio-distance==0.3.4 pandas scipy
```

Two environment notes that affect reproduction:

- The model is loaded with **eager attention**. The scaled dot-product
  attention path raised shape-broadcasting errors under these library
  versions.
- Batch size is 1 with 4-step gradient accumulation, chosen for T4 memory
  limits. A GPU with more memory would allow a larger true batch and may
  change training dynamics.

A full training sweep across all three ranks takes roughly 1–2 hours on a T4,
depending on how early stopping fires.

---

## 5. How to run

1. Open `MSc814_YouWang.ipynb` in Google Colab.
2. Set the runtime: **Runtime → Change runtime type → T4 GPU**.
3. Obtain CCOM-HuQin from the source in §2 and upload or mount the archive to
   the Colab environment.
4. **Runtime → Run all.** The notebook is sequential and self-contained.

It will extract the erhu subset, preprocess the audio, split at piece level,
train adapters at ranks 4/8/16 with early stopping, generate matched samples
from the baseline and each adapter, and compute FAD in two stages. Everything
is written to a fresh timestamped directory under `erhu_lora_runs/`.

Fixed seeds are used for partitioning (global seed 42), for each rank's
training run (42 + rank), and for each generated sample, so a clean re-run
reproduces the reported numbers.

---

## 6. Pipeline detail

**Preprocessing.** Channels averaged to mono, resampled to 32 kHz to match
MusicGen's EnCodec front end, peak-normalised with 0.1 dB headroom. Recordings
are cut into non-overlapping 30-second segments; tails shorter than 20 seconds
are discarded and 50 ms fades applied at boundaries.

**Canonical-piece splitting.** Two leakage risks are addressed. Segmenting
before partitioning would place near-identical segments on both sides of a
split. Less obviously, the corpus contains multi-part exports of single
performances (`-part1`, `-part2`) and, in some cases, different performances
of the same titled piece — so a file-level split would still separate related
material. Each file is therefore mapped to a canonical piece identifier by
stripping any trailing part index and normalising the remainder; files sharing
an identifier always land in the same partition. The 21 files reduce to 17
pieces, split 13/1/3, yielding 33 training, 3 validation and 6 test segments.

**Training.** LoRA with α = 2r, dropout 0.05, no bias adaptation, targeting
`q_proj` and `v_proj`. Because both self-attention and cross-attention use
these names and PEFT matches by name, 96 projections are adapted across the 24
decoder layers. AdamW, learning rate 5e-5, cosine schedule with 5% warm-up,
gradient clipping at 1.0, at most 10 epochs with early stopping (patience 2).

**Generation.** 20 samples per condition, same prompt, 400 max new tokens
(≈ 8 s, matching the reference crops), sampling on, temperature 1.0. All RNGs
are re-seeded to the same value before the corresponding sample under each
condition. Note that matched seeds do *not* make two samples renderings of the
same passage — the models induce different output distributions — so the
pairing removes a variance source without implying sample-for-sample
correspondence.

**Evaluation.** FAD with VGGish embeddings against 8-second centre crops:
three validation crops select the rank, six test crops are used once for the
reported comparison. Validation FAD is treated throughout as a selection
signal, never as a result.

---

## 7. Known limitations

Stated here so the artefacts are read with the right expectations; the
dissertation discusses each at length.

- **Reference sets are very small.** Three crops from one piece cannot
  represent the erhu as a class; six crops are better but still inadequate. An
  identical set of baseline generations scores ≈16.65 against the validation
  references and ≈6.90 against the test references — the difference is
  entirely a property of the reference sets, not the models.
- **No variance estimate.** One training run per rank and one generation batch
  per condition, so there is no way to say whether a 0.34% difference is
  smaller or larger than seed-to-seed variation.
- **One embedding.** FAD was introduced as a measure of perceived audio
  quality, validated against artificial distortions rather than instrument
  timbre, and its perceptual validity is strongly embedding-dependent. A low
  FAD here means proximity to the reference distribution in VGGish space,
  which is an imperfect proxy for bowing-level detail.
- **Peak, not loudness, normalisation.** Equalises maximum amplitude but not
  perceived loudness; VGGish embeddings are not level-invariant.

---

## 8. Citation

If this material is useful, please cite the CCOM-HuQin dataset (§2) and the
MusicGen paper:

> Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y.
> and Défossez, A. (2023) 'Simple and controllable music generation',
> *NeurIPS 36*. arXiv:2306.05284

Contact: bb24306@qmul.ac.uk
