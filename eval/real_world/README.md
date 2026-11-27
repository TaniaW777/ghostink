# Real-world test set

A small, hand-made test set of texts the model has never seen, from sources unlike the
training data. The goal is to measure how the detector behaves outside its training
distribution.

## Format: `real_world.csv`

UTF-8, comma-separated, one row per text, header `text,label,source,notes`.

| Column   | Required | Allowed values | Example |
|----------|----------|----------------|---------|
| `text`   | yes | the full text; keep line breaks inside quotes | `"Dear Ms. Smith, ..."` |
| `label`  | yes | `human` or `ai` | `human` |
| `source` | recommended | where the text comes from, `kind:detail` | `own:email`, `gutenberg:1`, `chatgpt:gpt-4o`, `gemini`, `mistral`, `llama` |
| `notes`  | optional | free text; tags separated by `;` | `paraphrased`, `pre-2022`, `translation`, `non-native` |

- A text with `paraphrased` in `notes` is counted as a paraphrased AI text.
- Texts under 50 words are kept but flagged: the model is not reliable on them.
- Tip: in Excel, use **Save As → CSV UTF-8 (comma delimited)**.

Check the file at any time:

```powershell
python -m human_or_ai.real_world
```

It reports empty rows, invalid labels, texts under 50 words, duplicates and missing sources,
then prints the number of usable texts per label and source. It exits with code 1 if
any row has an error.

## Suggested composition (at least 60 texts)

| Group | Count | Label | Notes tag |
|-------|-------|-------|-----------|
| Your own English writing (homework, emails, posts) | 15 | human | `non-native` if relevant |
| Texts published before November 2022 | 15 | human | `pre-2022` |
| AI texts (ChatGPT, Gemini, Mistral, Llama, about 4 each, varied topics) | 15 | ai | |
| The same AI texts rewritten by hand | 15 | ai | `paraphrased` |

Aim for 150 to 300 words per text so the results can be compared with the length
analysis in `reports/diagnosis.md`.

## Pre-2022 human sources

Public-domain sources. **The Project Gutenberg ebook numbers below were not verified
online** (gutenberg.org was unavailable when this list was written). Open each link
and check the title and translator before using it. The search links always work.

| Text | Why it is useful | Ebook link (check) | Search link |
|------|------------------|--------------------|-------------|
| US Declaration of Independence (1776) | formal, famously flagged by commercial detectors | https://www.gutenberg.org/ebooks/1 | https://www.gutenberg.org/ebooks/search/?query=declaration+of+independence |
| Lincoln, Gettysburg Address (1863) | short, formal speech | https://www.gutenberg.org/ebooks/4 | https://www.gutenberg.org/ebooks/search/?query=gettysburg+address |
| US Constitution (1787) | legal, very regular style | https://www.gutenberg.org/ebooks/5 | https://www.gutenberg.org/ebooks/search/?query=united+states+constitution |
| The Federalist Papers (1787-88) | formal argument, long sentences | https://www.gutenberg.org/ebooks/1404 | https://www.gutenberg.org/ebooks/search/?query=federalist+papers |
| Caesar, *The Gallic Wars* (English translation) | Roman, translated, plain narrative | https://www.gutenberg.org/ebooks/10657 | https://www.gutenberg.org/ebooks/search/?query=caesar+gallic |
| Cicero, *On Friendship* / *On Old Age* (English translation) | Roman, translated, essay style | https://www.gutenberg.org/ebooks/2808 | https://www.gutenberg.org/ebooks/search/?query=cicero |
| Marcus Aurelius, *Meditations* (English translation) | Roman, translated, reflective | https://www.gutenberg.org/ebooks/2680 | https://www.gutenberg.org/ebooks/search/?query=meditations+marcus+aurelius |
| Pliny the Younger, *Letters* (English translation) | Roman, translated, personal letters | https://www.gutenberg.org/ebooks/2811 | https://www.gutenberg.org/ebooks/search/?query=pliny+letters |
| Suetonius, *The Lives of the Twelve Caesars* (English translation) | Roman, translated, biography | https://www.gutenberg.org/ebooks/6400 | https://www.gutenberg.org/ebooks/search/?query=suetonius |
| Tacitus, *Agricola* / *Germania* (English translation) | Roman, translated, history | not checked | https://www.gutenberg.org/ebooks/search/?query=tacitus |
| Seneca, essays or letters (English translation) | Roman, translated, moral essay | not checked | https://www.gutenberg.org/ebooks/search/?query=seneca |
| Darwin, *On the Origin of Species* (1859) | formal scientific prose, close to the abstracts domain | https://www.gutenberg.org/ebooks/1228 | https://www.gutenberg.org/ebooks/search/?query=origin+of+species |
| Austen, *Pride and Prejudice* (1813) | literary, dialogue | https://www.gutenberg.org/ebooks/1342 | https://www.gutenberg.org/ebooks/search/?query=pride+and+prejudice |

Other pre-2022 human sources:

- **US federal government texts** (public domain in the US): Supreme Court opinions,
  NASA or NIH pages, dated before November 2022. Formal, modern English.
- **Old Wikipedia revisions**: open an article's *View history*, pick a revision dated
  before 30 November 2022, and copy its permanent link (`...&oldid=...`) into `source`.
  Wikipedia text is CC BY-SA 4.0, so it needs attribution and stays under that license.

## Licensing and privacy

This CSV is meant to be committed to a public repository.

- Gutenberg texts are public domain in the US. Copy only the excerpt, without the
  Project Gutenberg header and license.
- For Wikipedia excerpts, keep the revision link in `source` (attribution).
- For your own texts, remove names, email addresses and anything personal.
