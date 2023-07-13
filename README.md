# TrojPrompt [[Paper](https://arxiv.org/pdf/2306.06815.pdf)]

This repository contains code for our paper "[TrojPrompt: A Black-box Trojan Attack on Pre-trained Language Models](https://arxiv.org/pdf/2306.06815.pdf)". 
In this paper, we propose TrojPrompt to, an automatic and black-box framework to effectively generate universal and stealthy
triggers and insert Trojans into hard prompts of pre-trained large language models.

## Overview
The Workflow of TrojPrompt.
![detector](https://github.com/UCF-ML-Research/TrojPrompt/blob/main/figures/overview.png)



## Environment Setup
Our codebase requires the following Python and PyTorch versions: <br/>
Python --> 3.11.3   <br/>
PyTorch --> 2.0.1   <br/>

## Usage
we have split the code in three part

1. PromptSeed/ : Prompt Seed Tuning
2. Trigger/ : Universal Trigger Optimization
3. ProgressiveTuning/ : Progressive Prompt Poisoning

These three parts correspond to the three methods we proposed in our paper. Please refer to the corresponding folder for more details.