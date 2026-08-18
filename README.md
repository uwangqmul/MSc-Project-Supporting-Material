# MusicGen-small Erhu LoRA Adaptation

Supporting material for the MSc project *Domain Adaptation of MusicGen for
Erhu Timbre Adaptation via LoRA Fine-tuning*
You Wang · 250090532 · bb24306@qmul.ac.uk · MSc Artificial Intelligence, Queen Mary University of
London · Supervisor: Dr Aidan Hogg 

> ### Large files are on Google Drive
>
> The CCOM-HuQin archive (2.49 GB), the extracted erhu audio, the preprocessed
> segments and the generated evaluation samples are too large for GitHub. The
> complete supporting material is here:
>
> **https://drive.google.com/drive/folders/17MMbfv8v0xCPAZMw0d1zTvnxvQlkz4OB**
>
> The notebook is written to run in **Google Colab** with that folder mounted
> from Drive — see §5.

---

## 1. What this project does

The project asks a single question: **can Low-Rank Adaptation (LoRA) shift a
pre-trained text-to-music model towards the timbre of the erhu, using the
small corpus realistically available for a traditional instrument?**

Large text-to-music models such as MusicGen are trained overwhelmingly on
Western music, so they hold general musical knowledge without a detailed
representation of instruments that are sparsely represented in their training
data. Retraining on erhu audio is not feasible — the corpus here is 21
recordings totalling about 22.7 minutes. LoRA offers an alternative: freeze the
backbone and train a small number of additional parameters (0.13–0.53% of the
model, depending on rank).

The notebook compares a frozen `facebook/musicgen-small` baseline against LoRA
adapters at ranks 4, 8 and 16, under matched prompt, generation settings and
random seeds, scored with Fréchet Audio Distance (FAD).

**The headline result is negative, and deliberately reported as such.**
Evaluation runs in two stages: rank is selected on a validation reference set,
then the selected adapter is scored once against a test reference set withheld
from that selection. On the validation references the rank-8 adapter scored
9.14% below the baseline; on the held-out test references it scored 0.34%
*above* it. With three validation crops, six test crops and one training run
per rank, the design cannot distinguish a modest true effect from run-to-run
variation, so no improvement is claimed. The contribution is methodological: a
leakage-aware, seed-matched pipeline in which an apparent gain measured during
model selection is re-tested on independent references rather than reported as
a result.

Two scope limits, stated up front because they shape everything else:

- **Timbre, not technique.** CCOM-HuQin ships playing-technique annotations,
  but they are not used anywhere in this project. A 30-second segment usually
  contains several techniques, and an annotated technique may occupy only a
  fraction of it, so pairing a segment with a prompt naming one technique
  would be weak and partly incorrect supervision. The scope is whether output
  moves towards erhu recordings *as a class*.
- **A single fixed prompt.** Every audio crop is paired with one description:
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
performances released by the Central Conservatory of Music, annotated with
musical and playing-technique information. This project uses **only the erhu
subset**: 21 usable audio files, 1360.1 seconds in total, held in
`originalmusic/` on Drive.

The dataset is not redistributed in this repository. A copy of the archive and
of the extracted erhu subset is provided in the Google Drive folder for
assessment purposes only. All rights remain with the dataset authors: please
cite them in any work building on this material, obtain the dataset from its
original source for any other purpose, and check the licence attached to the
original release before redistributing any part of it.

The exact files used, their grouping into canonical pieces and the resulting
partition are recorded in `artifacts/split_manifest.json` and
`artifacts/source_audio_audit.csv` of each run, so the split can be
reconstructed without guesswork.

---

## 3. Notebook structure

`MSc814_YouWang.ipynb` runs top to bottom in ten sections:

| § | Section | What it does |
|---|---|---|
| 1 | Environment | Pins package versions and imports; sets device and seeding helper |
| 2 | Configuration and Drive paths | Mounts Drive, defines `Config`, creates a timestamped run directory, writes `config.json` |
| 3 | Canonical-piece split and duration audit | Groups files by canonical piece, splits deterministically, writes the split manifest and duration audit |
| 4 | Preprocessing without cross-split leakage | Mono, 32 kHz, peak normalisation, 30 s segmentation with fades; writes `segment_manifest.csv` |
| 5 | Dataset and deterministic evaluation crops | `ErhuDataset`: random 8 s crops for training, centre crops for validation and test |
| 6 | MusicGen labels, LoRA construction and training loop | On-the-fly EnCodec labels, LoRA injection, training with early stopping |
| 7 | Rank sweep | Trains ranks 4, 8 and 16; writes `rank_training_summary.csv` |
| 8 | Matched generation and duration-matched references | 20 samples per condition under shared seeds; exports 8 s reference crops |
| 9 | Validation selection, then one held-out test comparison | Validation FAD selects the rank; test FAD is computed once |
| 10 | Qualitative inspection panel | Matched-pair playback and mel spectrograms |

A closing **reporting checklist** cell lists what must exist before any number
is transferred into the dissertation.

The two archived-result cells near the top preserve values from an earlier
executed run for traceability; they are not outputs of the cells below them.

---

## 4. Output structure

Every execution creates a fresh timestamped directory. `RUN_ID` is a UTC
timestamp of the form `20250817T183000Z`.

```
MScProject/                            (CFG.project_root on Drive)
├── originalmusic/                     Erhu source audio (input)
└── erhu_lora_runs/
    └── <RUN_ID>/
        ├── artifacts/
        │   ├── config.json                 Full Config dataclass as run
        │   ├── source_audio_audit.csv      Per-file duration, rate, channels, split
        │   ├── split_manifest.json         Canonical piece IDs → split
        │   ├── segment_manifest.csv        Every 30 s segment and its source file
        │   ├── rank_training_summary.csv   One row per rank
        │   ├── validation_fad.csv          Baseline + all three ranks
        │   ├── test_fad.csv                Baseline + selected rank only
        │   ├── final_summary.json          Selected rank, both test FADs, deltas
        │   └── melspectrograms/            Saved spectrogram figures
        ├── dataset/
        │   ├── train/  val/  test/         Preprocessed 30 s segments
        ├── adapters/
        │   └── rank_{4,8,16}/
        │       ├── best/                   Best-validation LoRA weights
        │       ├── history.csv             Per-epoch train and validation loss
        │       └── target_modules.json     The 96 adapted projections, param counts
        └── evaluation/
            ├── references_val_8s/          3 centre crops
            ├── references_test_8s/         6 centre crops
            ├── generated_base/             20 baseline samples
            └── generated_rank_{4,8,16}/    20 samples per adapter
```

`target_modules.json` is worth opening: it enumerates every module that
received LoRA weights, which is how the claim that **both** self-attention and
cross-attention were adapted (96 projections across 24 decoder layers) is
verified rather than assumed.

---

## 5. Environment and how to run

Developed and run in **Google Colab** on a single **NVIDIA T4 GPU**. The first
cell installs everything; no local setup is required.

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11 |
| Transformers | 4.46.3 |
| PEFT | 0.13.2 |
| frechet-audio-distance | 0.3.4 |
| Base checkpoint | `facebook/musicgen-small` |

```bash
pip install "transformers==4.46.3" "peft==0.13.2" "accelerate>=0.34,<2" \
    soundfile sentencepiece "frechet-audio-distance==0.3.4" pandas scipy
```

**Steps**

1. Copy the Drive folder (or at least `originalmusic/`) into your own Drive at
   `MyDrive/MScProject/` — this is `CFG.project_root`, change it if you put it
   elsewhere.
2. Open `MSc814_YouWang.ipynb` in Colab and set **Runtime → Change runtime
   type → T4 GPU**.
3. **Runtime → Run all.** Section 2 will ask you to authorise Drive access.
4. Section 7 is the expensive cell. For a smoke test, set `CFG.ranks` to a
   one-element tuple first.

Fixed seeds throughout: global seed 42 for partitioning, `42 + rank` for each
training run, and `42 + 1000 + index` for each generated sample, so a clean
re-run reproduces the reported numbers.

Two environment notes that affect reproduction:

- The model is loaded with **eager attention** (`force_eager_attention`). The
  scaled dot-product attention path raised shape-broadcasting errors under
  these library versions.
- Batch size is 1 with 4-step gradient accumulation, set by T4 memory limits. A
  larger GPU would allow a bigger true batch and may change training dynamics.

---

## 6. Method summary

**Canonical-piece splitting.** Two leakage risks are addressed. Segmenting
before partitioning would place near-identical segments on both sides of a
split. Less obviously, the corpus contains multi-part exports of single
performances (`-part1`, `-part2`) and, in some cases, different performances of
the same titled piece, so even a file-level split would separate related
material. `canonical_piece_id()` strips any trailing part index and normalises
the remainder to lower-case alphanumeric tokens; files sharing an identifier
always land in the same partition. The 21 files reduce to 17 pieces, split
13/1/3, yielding 33 training, 3 validation and 6 test segments.

**Preprocessing.** Channels averaged to mono, resampled to 32 kHz to match
MusicGen's EnCodec front end, peak-normalised with 0.1 dB headroom, cut into
non-overlapping 30-second segments. Tails under 20 seconds are discarded; 50 ms
fades are applied at boundaries.

**Training.** LoRA with α = 2r, dropout 0.05, no bias adaptation, targeting
`q_proj` and `v_proj`. Because both self-attention and cross-attention use
those names and PEFT matches by name, 96 projections are adapted across the 24
decoder layers. Labels are built on the fly by encoding each crop with the
model's own EnCodec encoder under `torch.no_grad`. AdamW, learning rate 5e-5,
cosine schedule with 5% warm-up, gradient clipping at 1.0, at most 10 epochs
with early stopping (patience 2).

**Generation.** 20 samples per condition: same prompt, 400 max new tokens
(≈ 8 s, matching the reference crops), sampling on, temperature 1.0. All RNGs
are re-seeded to the same value before the corresponding sample under each
condition. Matched seeds do *not* make two samples renderings of the same
passage — the models induce different output distributions — so the pairing
removes a variance source without implying sample-for-sample correspondence.

**Evaluation.** FAD with VGGish embeddings (16 kHz internally, PCA and
activation statistics disabled) against 8-second centre crops: three validation
crops select the rank, six test crops are used once for the reported
comparison. Validation FAD is treated throughout as a selection signal, never
as a result.

---

## 7. Known limitations

Stated here so the artefacts are read with the right expectations; the
dissertation discusses each at length.

- **Reference sets are very small.** Three crops from one piece cannot
  represent the erhu as a class; six crops are better but still inadequate. An
  identical set of baseline generations scores ≈16.65 against the validation
  references and ≈6.90 against the test references — that gap is entirely a
  property of the reference sets, not the models.
- **No variance estimate.** One training run per rank and one generation batch
  per condition, so there is no way to say whether a 0.34% difference is
  smaller or larger than seed-to-seed variation.
- **One embedding.** FAD was introduced as a measure of perceived audio
  quality, validated against artificial distortions rather than instrument
  timbre, and its perceptual validity is strongly embedding-dependent. A low
  FAD here means proximity to the reference distribution in VGGish space, an
  imperfect proxy for bowing-level detail.
- **Section 10 is exploratory.** The playback and spectrogram panel is informal
  inspection by the author, not a listening study — no rating protocol, no
  blinding, no independent listeners.
- **Peak, not loudness, normalisation.** Equalises maximum amplitude but not
  perceived loudness; VGGish embeddings are not level-invariant.

---

## 8. Citation

Please cite the CCOM-HuQin dataset (§2) and the MusicGen paper:

> Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y.
> and Défossez, A. (2023) 'Simple and controllable music generation',
> *NeurIPS 36*. arXiv:2306.05284
