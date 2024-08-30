# Energy-Based Model (EBM) Comparison: Contrastive Loss vs. Sliced Score-Based Loss

This repository contains code and resources for training and comparing Energy-Based Models (EBM) using contrastive loss and sliced score-based loss. The project explores the implementation of these losses and compares their performance on various datasets.

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Contrastive Loss](#contrastive-loss)
  - [Sliced Score-Based Loss](#sliced-score-based-loss)
- [Experiments](#experiments)
- [Results](#results)

## Introduction

Energy-Based Models (EBMs) are a class of generative models that model the data distribution by learning an energy function. The energy function assigns low energy (high probability) to data samples and high energy (low probability) to non-data samples.

## Useage
The code is made on a jupiter notebook.

## Methodology
In this project, we explore and compare the performance of two loss functions for training EBMs:
- **Contrastive Loss**: A widely used approach in which the model learns to distinguish between positive and negative samples using a semi-unsupervised learning in the sense that there aren't any target variables.
- **Sliced Score-Based Loss**: A loss function that leverages sliced Wasserstein distances and score matching techniques. In this method the score based loss is converted into it's second order form containing hessian of the energy function. This hessian is approximated by a using random projections. This method is equivalent to matching the random projections of score of model and the actual density

The goal is to analyze the effectiveness of these loss functions and understand their impact on the model's ability to learn meaningful representations.

## Code Structure

```plaintext
├── data/                     # Datasets used in the experiments borrowed from https://github.com/kamenbliznashki/normalizing_flows
├── models/                   # EBM model architectures
├── Trainer/
│   ├── contrastive_loss      # Implementation of contrastive loss
│   ├── sliced_loss           # Implementation of sliced score-based loss
│   ├── trainer               # Training loop and utilities
│   └── utils                 # Helper functions for data loading and preprocessing
├── results                   # Visualisation of samples and energy potentials
