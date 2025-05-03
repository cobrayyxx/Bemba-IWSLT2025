# Collection of Code for IWSLT 2025, Low Resource Track, Bemba-English

This repo is a collection of a bunch of codes/notebooks that are purposed to create a speech-translation systems, Bemba to English. 
The directory structure of this repo is like below.

```
.
├── data-related_notebooks
├── evaluation_notebooks
├── files
├── other_notebooks
├── training_notebooks
├── readme.md
└── .gitignore
```
## `data-related_notebooks` directory
This repository is consists of some notebooks that are related to preprocessing and exploration of the data, like doing EDA for dataset, augmenting dataset, etc.

## `evaluation_notebooks` directory
This repository contains several notebooks that are related to evaluating the model/systems. The models/systems that we evaluated are machine-translation(MT), The whole Cascaded Systems, and The whole End-to-End Systems. BLEU, ChrF++, and AfriCOMET(COMET) are the metric evaluation that we used. For the test-set, we used `devtest` split from FLORES-200 and `test` split from Big-C.

## `files` directory

In files directory contains several files, most of them are the augmented dataset files, the transcription and translation files.

## `other_notebooks` directory

This directory have two notebook files inside of it. One notebook is for predicting the test-set from the IWSLT organizer and the other is for find the best hyperparameter values using optuna.

## `training_notebooks` directory

All of the training scripts and notebooks are placed in this directory. Starting from training machine-translation models, transcription/automatic speech recognition (ASR) model, to training End-to-End model. There are several notebook files in here, some of them were used for training the same base-model with different dataset and some of them were trained on the different base-model.

