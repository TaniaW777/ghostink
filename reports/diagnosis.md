# Diagnosis of the v1 baseline

v1 is the model from the original notebook: `TextVectorization` → `Embedding(10000, 16, mask_zero=True)`
→ `Dropout(0.2)` → `GlobalAveragePooling1D` → `Dropout(0.2)` → `Dense(1, sigmoid)`, 160,017 parameters.
All numbers below were measured with seed 42 and a decision threshold of 0.5. Label 1 = AI.

Reproduce:

```powershell
python -m human_or_ai.train --version v1      # about 3 minutes on CPU
python -m human_or_ai.diagnose --version v1   # about 9 minutes (includes 2 cross-dataset trainings)
```

Raw tables: [`diagnosis/v1/tables.md`](diagnosis/v1/tables.md) and the JSON files next to it.
Walk-through with comments: [`notebooks/02_error_analysis.ipynb`](../notebooks/02_error_analysis.ipynb).

## Summary

| Evaluation | Texts | Accuracy | FPR human ↓ | TPR AI ↑ | ROC-AUC | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Test, in-distribution (both datasets) | 9,758 | 96.0% | 5.3% | 97.6% | 0.990 | 0.018 |
| Cross: train *recent* → test *abstracts* | 5,727 | 53.9% | 10.4% | 18.3% | 0.648 | 0.341 |
| Cross: train *abstracts* → test *recent* | 4,031 | 59.7% | 3.7% | 0.8% | 0.468 | 0.389 |
| Real-world set | - | pending | pending | pending | pending | pending |

For reference, the same cross-dataset models score 95.1% (*recent* → *recent*) and 99.5%
(*abstracts* → *abstracts*) on their own test split.

FPR human = share of human texts flagged as AI (the costly error). TPR AI = share of AI texts
detected. ECE = expected calibration error, 10 equal-width bins of P(AI).

**In one sentence:** v1 recognizes the two sources it was trained on very well, but it barely
transfers from one to the other. It relies on formatting artifacts, and when it is wrong it is
often very sure of itself.

## Findings

### 1. It does not generalize across datasets

Trained on one dataset and tested on the other, v1 drops to near chance (AUC 0.65 and 0.47).
Trained on abstracts, it labels almost every recent text as human: it catches 13 of 1,546 AI
answers. The 96% test accuracy measures recognition of *familiar sources*, not detection of
AI writing in general. This is the failure mode reported by the RAID benchmark (Dugan et al.,
ACL 2024).

### 2. Formatting artifacts separate the classes

Counted on the 47,085 training texts ([table](diagnosis/v1/tables.md#shortcut-patterns-train-set-per-class)):

| Pattern | Human texts | AI texts | AUC alone | Points to |
|---|---:|---:|---:|---|
| Split `n't` (`do n't`) | 21.0% | 0.004% | 0.61 | human |
| Split `'s` (`it 's`) | 23.9% | 0.02% | 0.62 | human |
| Split `'re` `'ll` `'ve` `'m` `'d` | 15.3% | 0% | 0.58 | human |
| Normal contraction (`don't`, `it's`) | 5.2% | 37.3% | 0.66 | AI |
| `URL_0` placeholder | 7.0% | 0.05% | 0.54 | human |
| Literal `\n` (backslash + n) | 0% | 7.3% | 0.54 | AI |
| Real line break | 0.7% | 58.4% | 0.79 | AI |
| Space before `, . ; : ! ? )` | 56.3% | 3.4% | 0.77 | human |
| Space after `(` | 21.2% | 0.05% | 0.61 | human |
| Mean length (words) | 162.9 | 172.1 | 0.58 | AI |

Each dataset has its own artifacts:

- *recent*: 88% of human texts have a space before punctuation (0.3% of AI texts), and split
  contractions and `URL_0` appear only in human texts. They come from the tokenized
  HC3/Reddit human answers.
- *abstracts*: **100% of AI abstracts contain a line break, 0% of human abstracts.** The AI
  abstracts start with blank lines.

v1 cannot see spaces or line breaks directly (its standardization removes punctuation and
splits on whitespace), but it sees their side effects as tokens, as shown below. A future
style branch measuring paragraphs or punctuation spacing would learn these artifacts directly.

### 3. The most influential words include artifacts

From the learned weights (embedding · output weights; [figure](figures/v1_top_words.png)):

- **Human side, artifacts:** `nt` (#1), `s` (#2), `re` (#3), `ll`, `ve`, `ca` (`ca n't`), `wo`
  (`wo n't`), `url0`, plus `ie`, `eg`, `etc`, `andor` (punctuation removed from `i.e.`, `e.g.`,
  `and/or`). The standardization turns `do n't` into the tokens `do` + `nt`.
- **AI side, artifacts:** `earths`, `bodys`, `cant`. The joined spellings `Earth's`, `body's`
  and `can't` become one token, and in this data mostly AI texts write them that way.
  **A person who writes `can't` correctly is pushed towards AI.**
- **Source format, not style:** `background`, `conclusions`, `supplementary`, `0001`
  (`p < 0.0001`) come from the headers and statistics of real structured abstracts; `edit`
  is a Reddit habit.
- **Plausible style signals:** human `i`, `basically`, `pretty`, `probably`, `got`; AI
  `presents`, `investigates`, `examines`, `explores`, `aims`, `seeks`, `paper`, `overall`,
  `additionally`, `comprehensive`, `insights`, `ultimately`, `important`.

This explains the real-life errors reported in the notebook: normal English with ordinary
contractions loses the strongest "human" signals and gains "AI" ones.

### 4. Short texts are a weak spot, and mostly human

| Length (words) | Texts | Human / AI | Accuracy | FPR human | ECE |
|---|---:|---:|---:|---:|---:|
| < 50 | 861 | 806 / 55 | 87.7% | 12.5% | 0.119 |
| 50-150 | 3,367 | 1,606 / 1,761 | 95.8% | 6.0% | 0.013 |
| 150-300 | 4,779 | 2,348 / 2,431 | 97.4% | 3.3% | 0.018 |
| > 300 | 751 | 586 / 165 | 97.5% | 1.9% | 0.021 |

Below 50 words, the false positive rate doubles and calibration breaks. The worst false
positives are human texts of 3 to 14 words scored at P(AI) = 1.00 (for example, "See related
letter").

**Length alone** (logistic regression on log word count) gives an AUC of 0.74 on *recent*, where
AI answers are longer, and 0.70 on *abstracts*, where AI abstracts are shorter (the AUC is
folded, so a signal in either direction counts). The two directions cancel when the datasets
are mixed (AUC 0.51), but a model can still use length once it recognizes the source.

### 5. Calibration: good on average, overconfident when wrong

In-distribution ECE is low (0.018), because most predictions sit near 0 or 1 and are correct.
But:

- 28.4% of the errors are made with more than 90% confidence; mean confidence when wrong: 0.77;
- between P(AI) = 0.1 and 0.8, the reliability curve sits below the diagonal: when v1 says
  "60% AI", the real share of AI texts is lower ([figure](figures/v1_reliability_test.png));
- out of distribution, ECE jumps to 0.34-0.39.

### 6. Validation leak through abstract titles

The abstracts validation split was cut at random from the train split, and 7,297 of the
11,042 training titles have both a human and an AI version. **82% of the abstracts used for
validation (3,738 of 4,581) share their title with a training abstract.** The official test
split has no overlap (0%), so the test score is not inflated by this leak. But early stopping
was driven by a partly leaked validation set. Phase 2 splits by title and will report the
before/after.

## Biases and shortcuts to fix

| # | Problem | Evidence | Planned fix |
|---|---|---|---|
| 1 | Split contractions mark "human", joined ones mark "AI" | `nt`, `s`, `re` = top 3 human words; `cant`, `earths` in the AI top 30 | Phase 2: re-join contractions in `normalize.py` |
| 2 | Spaces around punctuation | 56% vs 3.4% of texts, AUC 0.77 | Phase 2: remove spaces before punctuation and after `(` |
| 3 | URL placeholder | `url0` in the human top 15 | Phase 2: one neutral token for `URL_0` and real URLs |
| 4 | Line breaks and literal `\n` | 100% vs 0% in abstracts; literal `\n` only in AI | Phase 2: convert literal `\n`, strip leading and trailing blank lines. No paragraph feature in v4 unless both classes keep their line breaks |
| 5 | Source format words (`background`, `conclusions`, `edit`) | human top 30 | Phase 2: investigate stripping section headers; keep as a known limit otherwise |
| 6 | Short texts | FPR 12.5% under 50 words, 94% of them human | Phase 3: train on 50-250-word excerpts; phase 4: abstain under 50 words |
| 7 | Length differs by source | AUC 0.74 / 0.70 inside each dataset | Phase 3: excerpts of controlled length; report length-only baseline per version |
| 8 | No transfer across datasets | AUC 0.65 / 0.47 | Track the cross-dataset scores for every version; select the final model on them and on the real-world set |
| 9 | Overconfident errors | 28% of errors above 90% confidence | Phase 3: label smoothing; phase 4: temperature scaling + "Uncertain" zone |
| 10 | Title leak in validation | 82% of abstract validation titles seen in train | Phase 2: split by title |

## Not measured yet

- **Real-world set**: waiting for `eval/real_world/real_world.csv` (at least 60 texts). Once it
  is filled in, `python -m human_or_ai.diagnose --version v1 --skip-cross` adds its line to the
  summary without retraining.
- **Near-duplicate texts** across splits (only exact duplicates are removed today): phase 2.

## v2: effect of normalization

v2 is the v1 architecture trained on normalized text (`configs/v2.yaml`: `normalize.py`, abstracts
validation split by title, near-duplicate removal). Reproduce:

```powershell
python -m human_or_ai.train --version v2 --config configs/v2.yaml
python -m human_or_ai.diagnose --version v2 --config configs/v2.yaml
python -m human_or_ai.compare --versions v1 v2
```

Content of [`comparison_v1_v2.md`](comparison_v1_v2.md) (generated by `compare.py`):

Auto-generated by `python -m human_or_ai.compare --versions v1 v2` from `reports/diagnosis/<version>/*.json`. Do not edit by hand.

Threshold 0.5, label 1 = AI. Cross lines: model retrained on one dataset only, tested on the other dataset's test split. `< 50 words`: test texts shorter than 50 words.

### Metrics

| Evaluation | Version | Accuracy | FPR human ↓ | TPR AI ↑ | ROC-AUC | ECE ↓ | Texts |
|---|---|---:|---:|---:|---:|---:|---:|
| test (in-distribution) | v1 | 96.0% | 5.3% | 97.6% | 0.990 | 0.018 | 9,758 |
| test (in-distribution) | v2 | 95.4% | 5.2% | 96.0% | 0.988 | 0.010 | 10,198 |
| cross: recent → abstracts | v1 | 53.9% | 10.4% | 18.3% | 0.648 | 0.341 | 5,727 |
| cross: recent → abstracts | v2 | 52.1% | 7.0% | 11.3% | 0.638 | 0.384 | 5,727 |
| cross: abstracts → recent | v1 | 59.7% | 3.7% | 0.8% | 0.468 | 0.389 | 4,031 |
| cross: abstracts → recent | v2 | 57.6% | 4.0% | 0.7% | 0.473 | 0.412 | 4,471 |
| test, texts < 50 words | v1 | 87.7% | 12.5% | 90.9% | 0.954 | 0.119 | 861 |
| test, texts < 50 words | v2 | 87.0% | 12.9% | 85.0% | 0.947 | 0.121 | 1,031 |

### Top 15 words

Most human and most AI words from the learned weights (`top_words.json`), rank 1 = strongest.

| Rank | v1 human | v2 human | v1 AI | v2 AI |
|---:|---|---|---|---|
| 1 | `nt` | `etc` | `presents` | `presents` |
| 2 | `s` | `basically` | `towards` | `examines` |
| 3 | `re` | `ie` | `investigates` | `overall` |
| 4 | `ie` | `supplementary` | `examines` | `towards` |
| 5 | `etc` | `conclusions` | `overall` | `findings` |
| 6 | `supplementary` | `isnt` | `findings` | `investigates` |
| 7 | `conclusions` | `background` | `important` | `important` |
| 8 | `basically` | `probably` | `additionally` | `seeks` |
| 9 | `background` | `chapter` | `evaluates` | `treatments` |
| 10 | `chapter` | `pretty` | `treatments` | `utilizing` |
| 11 | `url0` | `andor` | `insight` | `additionally` |
| 12 | `ca` | `theyre` | `comprehensive` | `explores` |
| 13 | `ll` | `eg` | `earths` | `located` |
| 14 | `probably` | `theres` | `bodys` | `managing` |
| 15 | `although` | `essentially` | `seeks` | `comprehensive` |

**Interpretation.**
The test sets are not the same: v2 is tested on 10,198 texts and v1 on 9,758, because the v2
data pipeline changes which texts are kept, so the 96.0% → 95.4% accuracy change is not a
like-for-like comparison.
Normalization lowers the test ECE from 0.018 to 0.010 and leaves the human false positive rate
almost unchanged (5.3% → 5.2%), while the AI detection rate drops from 97.6% to 96.0%.
It does not fix the transfer problem: the cross-dataset AUC stays near chance (0.648 → 0.638
for recent → abstracts, 0.468 → 0.473 for abstracts → recent), and texts under 50 words keep a
false positive rate of 12.9%.
The split-contraction artifacts `nt`, `s`, `re`, `ll`, `ca` and `url0` leave the 15 most human
words, but re-joined contractions (`isnt` at rank 6, `theyre` at 12, `theres` at 13) now point
to human, and source-format words such as `supplementary` (rank 4) and `conclusions` (rank 5)
remain.
