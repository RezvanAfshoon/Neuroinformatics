# Neural Spike Train Classification: Familiar vs. Novel Stimuli

Supervised ML pipeline for decoding stimulus familiarity from neural spike train recordings, developed as a course project for the Neuroinformatics course at Osnabrück University (graded 1.0).

## Overview

Given multi-neuron spike train recordings across trials, the goal is to classify whether the presented stimulus was familiar or unfamiliar.

## Pipeline

1. **Data loading**  spike train arrays (trials × neurons × timesteps) loaded from `.npy` files
2. **Visualization**  raster plots and spike count distributions for exploratory analysis
3. **Feature extraction**  two neural coding strategies:
   - *Rate coding*: average firing rate per neuron per trial (one feature vector per trial)
   - *Population total spikes*: total spike count across all neurons and timesteps
4. **Exploratory analysis**  KL Divergence between familiar and unfamiliar firing rate distributions (KL = 4.53, indicating strong class separation)
5. **Classification**  Logistic Regression and Random Forest trained on extracted features
6. **Model comparison**  effect of feature dimensionality (all neurons vs. 100 neurons) and classifier type on accuracy
