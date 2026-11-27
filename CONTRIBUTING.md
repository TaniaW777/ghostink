# Contributing to Ghostink

Thank you for your interest in this project. Bug reports, questions, ideas and pull requests
are all welcome.

## Set up

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

```bash
# macOS / Linux (bash)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the checks

The same checks run in CI (`.github/workflows/ci.yml`) on every push and pull request.

```bash
ruff check .
pytest -q
```

Tests never download data and never train a full model, so they run in seconds on a CPU.

## Open an issue

Open an issue on [GitHub](https://github.com/TaniaW777/ghostink/issues) to report a bug,
ask a question or propose an idea. For a bug, include the command you ran, the full error
message, your operating system and your Python version.

## Open a pull request

1. Fork the repository and create a branch from `main`.
2. Make your change, with a test when the change touches code in `src/human_or_ai/`.
3. Run `ruff check .` and `pytest -q`: both must pass.
4. Open a pull request that explains what changed and why. If a result changed, say which
   command produces it.

## Project rules

- **No pre-trained models.** No transformers, BERT, GPT, GloVe, word2vec, fastText, pre-trained
  tokenizers, detection APIs, or language models used as judges or to compute a perplexity.
  Every weight, vocabulary and threshold is learned from the project's data. Hugging Face is used
  only to load datasets.
- **Every number must be reproducible with a command.** A number that appears in a report or in
  the README must come from a file written by the code (`reports/*.json` or a generated
  `reports/*.md`), and the pull request must give the command that regenerates it. Generated
  reports are never edited by hand.
- Files are UTF-8 without BOM, with LF line endings.

## Good first issues

Taken from the *Future work* section of the [README](README.md):

1. **Real-world test set**: add texts to `eval/real_world/real_world.csv` following
   [`eval/real_world/README.md`](eval/real_world/README.md), then check the file with
   `python -m human_or_ai.real_world`.
2. **N-gram branch**: add a TF-IDF n-gram baseline trained from scratch and report it next to v2.
3. **Stylometric branch**: add simple style features (sentence length, punctuation, vocabulary
   richness) and measure what they add.
4. **Temperature calibration**: fit a temperature on the validation set and report the change in
   calibration error.
5. **Cross-dataset thresholding**: check whether decision thresholds calibrated on one dataset
   transfer to another.
