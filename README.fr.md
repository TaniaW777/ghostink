# Ghostink

*Un détecteur neuronal de texte généré par IA, construit de zéro, et un regard honnête sur les raisons pour lesquelles ce problème est si difficile.*

[![CI](https://github.com/TaniaW777/ghostink/actions/workflows/ci.yml/badge.svg)](https://github.com/TaniaW777/ghostink/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)

*English version: [README.md](README.md). La démo, la pile technique, les données, le
fonctionnement du modèle, les pistes futures, la contribution et la licence sont décrits dans
le README anglais.*

## Pourquoi ce projet

Les textes générés par IA sont désormais partout : dans les devoirs, les candidatures, les
articles et les publications sur les réseaux sociaux. Les enseignants veulent protéger
l'intégrité académique, et les lecteurs veulent savoir quand une désinformation est produite
en masse par une machine. Les watermarks ne suffisent pas, car seule l'entreprise qui a créé un
modèle peut lire son watermark, et les textes des autres modèles n'en portent aucun. Ce projet
pose une question plus simple : qu'est-ce qu'un petit modèle, entraîné de zéro, peut vraiment
apprendre de la différence entre l'écriture humaine et celle d'une IA ?

## Construit de zéro

Aucun modèle pré-entraîné n'est utilisé : pas de transformer, pas d'embeddings pré-entraînés,
pas d'API de détection. Le vocabulaire, les embeddings de mots et les seuils de décision sont
tous appris sur les données du projet. Hugging Face sert uniquement à télécharger les jeux de
données. Le projet a commencé comme un exercice de cours, adapté d'un notebook d'analyse de
sentiment IMDB.

## Résultats

Seuil 0,5, étiquette 1 = IA. Généré par `python -m human_or_ai.compare --versions v1 v2`
([`reports/comparison_v1_v2.md`](reports/comparison_v1_v2.md)). v1 est le modèle d'origine du
notebook, sur le texte brut ; v2 est la même architecture sur du texte normalisé et
dédoublonné. Lignes croisées : modèle réentraîné sur un seul jeu de données, testé sur le
split de test de l'autre jeu. Le tableau est reproduit tel quel (en anglais).

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

FPR human = part des textes humains signalés comme IA (l'erreur coûteuse). TPR AI = part des
textes IA détectés. ECE = erreur de calibration attendue.

Avec les seuils de décision de v2 (t_human = 0.2396, t_ai = 0.8133, au moins 50 mots), générés
par `python -m human_or_ai.decision_report`
([`reports/decision_v2.md`](reports/decision_v2.md)). Couverture (*coverage*) = part des
textes qui reçoivent un verdict Humain ou IA. Le FPR, le FNR et l'accuracy sont calculés sur
les textes décidés uniquement.

| evaluation | n | coverage | uncertain | not enough text | FPR (decided) | FNR (decided) | accuracy (decided) |
|---|---|---|---|---|---|---|---|
| test v2 | 10198 | 82.8% | 7.1% | 10.1% | 1.4% | 1.8% | 98.4% |

**Ce que disent ces chiffres.**
Sur son propre test, v2 atteint 95.4 % d'accuracy et une ROC-AUC de 0.988, mais v1 et v2 ne
sont pas testés sur les mêmes textes (9,758 contre 10,198) : la baisse depuis 96.0 % n'est donc
pas une comparaison à l'identique.
La normalisation améliore la calibration : l'ECE du test passe de 0.018 à 0.010, avec un taux
de faux positifs humains presque inchangé (5.3 % → 5.2 %).
Elle ne règle pas la généralisation : entraîné sur un jeu de données et testé sur l'autre, v2
reste proche du hasard, avec une ROC-AUC de 0.638 et 0.473.
Les textes courts restent un point faible : sous 50 mots, 12.9 % des textes humains sont
signalés comme IA, c'est pourquoi l'application les refuse.
Avec les seuils, le taux de faux positifs sur les textes décidés tombe à 1.4 %, mais seuls
82.8 % des textes du test reçoivent un verdict : 7.1 % sont *Incertains* et 10.1 % sont trop
courts.

## Ce que j'ai découvert : l'apprentissage par raccourcis

Le premier modèle (v1) obtenait **96.0 %** sur son test. Ce chiffre semblait excellent, et il
était trompeur. Le diagnostic complet est dans [`reports/diagnosis.md`](reports/diagnosis.md).

**Il reconnaissait des sources, pas l'écriture d'une IA.** Entraîné sur *abstracts* et testé
sur *recent*, v1 n'a détecté que 13 réponses IA sur 1,546. D'un jeu de données à l'autre, sa
ROC-AUC tombait à 0.648 et 0.468, proche d'un pile ou face. Les 96.0 % mesuraient sa capacité à
reconnaître deux sources familières.

**L'artefact `do n't`.** Les réponses humaines de *recent* viennent de corpus tokenisés où les
contractions sont séparées : `do n't`, `it 's`. Dans le jeu d'entraînement, 21.0 % des textes
humains contenaient un `n't` séparé, contre 0.004 % des textes IA, alors qu'une contraction
normale (`don't`, `it's`) apparaissait dans 5.2 % des textes humains et 37.3 % des textes IA.
Le modèle l'a appris : `nt` était son mot « humain » numéro 1, et `cant` (issu de `can't`)
figurait parmi ses premiers mots « IA ». **Une personne qui écrit `can't` correctement était
poussée vers l'IA.**

**Les sauts de ligne dans les résumés IA.** Dans *abstracts*, 100 % des résumés IA contenaient
un saut de ligne, contre 0 % des résumés humains : les résumés IA commencent par des lignes
vides. Sur l'ensemble du jeu d'entraînement, un vrai saut de ligne apparaissait dans 58.4 % des
textes IA et 0.7 % des textes humains, et ce seul indice atteignait une AUC de 0.79.

**Après normalisation (v2).** `normalize_text` est appliquée aux deux classes. Dans le jeu
d'entraînement v2 ([`reports/shortcuts_v2.json`](reports/shortcuts_v2.json)), les contractions
séparées, les marqueurs d'URL, les sauts de ligne et les espaces avant la ponctuation tombent
tous à 0.0 % dans les deux classes, avec une AUC seule de 0.5. Les contractions jointes
apparaissent désormais dans 44.640 % des textes humains et 40.448 % des textes IA (AUC 0.531).
La longueur du texte reste un indice (AUC 0.605 pour le nombre de mots seul). Retirer ces
artefacts n'a pas permis au modèle de généraliser (voir les lignes croisées dans
[Résultats](#résultats)) : l'écart restant est dans les données elles-mêmes.

## Limites

- **Anglais uniquement.**
- **Deux sources de données seulement.** Le modèle a vu deux jeux de données, pas la variété
  des textes réels.
- **Les textes de moins de 50 mots sont refusés** (*Not enough text*) : sur eux, 12.9 % des
  textes humains sont signalés comme IA.
- **Faible généralisation entre domaines** : entraîné sur un jeu de données et testé sur
  l'autre, la ROC-AUC est de 0.638 (*recent* → *abstracts*) et 0.473 (*abstracts* → *recent*).
- **Non testé sur des textes IA paraphrasés**, ni sur des textes de modèles d'IA absents des
  données d'entraînement.
- **Le taux de faux positifs n'est pas nul** : sur les textes du test décidés, il est de 1.4 % ;
  l'outil peut donc accuser un humain.
- **Biais probable contre les personnes qui écrivent en anglais sans être natives**, signalé
  pour les détecteurs d'IA en général par Liang et al. (2023). Il n'a pas été mesuré pour ce
  modèle.
- **Le modèle ignore l'ordre des mots** : la moyenne des embeddings transforme chaque texte en
  sac de mots.

## Éthique

Cet outil donne une alerte, jamais une preuve. Son verdict peut être faux, dans les deux sens,
et il ne doit jamais servir à sanctionner, à baisser une note ni à accuser qui que ce soit. Si
un texte fait naître un doute, la bonne étape suivante est une conversation avec son auteur,
pas une sanction.

## Questions ouvertes

Si les entreprises qui construisent ces modèles ne savent pas détecter de façon fiable leurs
propres productions (OpenAI a retiré son classifieur de texte IA en juillet 2023 à cause de sa
faible précision : il détectait 26 % des textes IA et signalait 9 % des textes humains), sur
quelle base accuse-t-on des étudiants et des auteurs d'avoir utilisé l'IA ?

Des textes écrits bien avant l'existence de l'IA sont parfois étiquetés comme générés : lors
d'un test, ZeroGPT a estimé le préambule de la Déclaration d'indépendance de 1776 généré par IA
à 97.93 %. Les détecteurs mesurent-ils le style d'une machine, ou simplement une écriture
formelle, régulière et sans faute ?

Sept détecteurs populaires ont signalé en moyenne 61.3 % des essais écrits par des personnes
non anglophones natives comme générés par IA (Liang et al., 2023). Qui porte la responsabilité
d'un faux positif : l'outil, l'institution qui l'utilise, ou personne ?

Les modèles de langage apprennent à écrire à partir de textes humains. À mesure qu'ils
progressent, existe-t-il encore une frontière stable entre le style « humain » et le style
« machine », ou la détection devra-t-elle passer du texte lui-même à sa provenance
(watermarks, traces du processus d'écriture) ?

Sources :
[OpenAI](https://openai.com/index/new-ai-classifier-for-indicating-ai-written-text) ·
[Decrypt](https://decrypt.co/286121/ai-detectors-fail-reliability-risks) ·
[Liang et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10382961/) ·
[arXiv:2405.07940](https://arxiv.org/abs/2405.07940)

## Lancer le projet en local

L'entraînement tourne sur CPU. La première exécution télécharge les deux jeux de données depuis
Hugging Face.

**Windows (PowerShell)**

```powershell
git clone https://github.com/TaniaW777/ghostink.git
cd ghostink
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Entraîner v2, le diagnostiquer et calculer les seuils de décision utilisés par l'application
python -m human_or_ai.train --version v2 --config configs/v2.yaml
python -m human_or_ai.diagnose --version v2 --config configs/v2.yaml
python -m human_or_ai.decision_report --config configs/v2.yaml

# Lancer l'application
python app/app.py

# Lancer les tests
pytest
```

**macOS / Linux (bash)**

```bash
git clone https://github.com/TaniaW777/ghostink.git
cd ghostink
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Entraîner v2, le diagnostiquer et calculer les seuils de décision utilisés par l'application
python -m human_or_ai.train --version v2 --config configs/v2.yaml
python -m human_or_ai.diagnose --version v2 --config configs/v2.yaml
python -m human_or_ai.decision_report --config configs/v2.yaml

# Lancer l'application
python app/app.py

# Lancer les tests
pytest
```

Pour reproduire la référence v1 et le tableau de comparaison, lancez aussi
`python -m human_or_ai.train --version v1`, `python -m human_or_ai.diagnose --version v1` et
`python -m human_or_ai.compare --versions v1 v2`. Pour analyser un seul texte en ligne de
commande : `python -m human_or_ai.predict "votre texte ici"`.
