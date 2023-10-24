# TrojLLM [[Paper](https://arxiv.org/pdf/2306.06815.pdf)]

This repository contains code for our paper "[TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models](https://arxiv.org/pdf/2306.06815.pdf)". 
In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy
triggers and inserts Trojans into hard prompts of pre-trained large language models.

## Overview
The Workflow of TrojLLM.
![detector](https://github.com/UCF-ML-Research/TrojLLM/blob/main/figures/overview.png)



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

## Citation
If you find TrojLLM useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{xue2023trojprompt,
  title={TrojPrompt: A Black-box Trojan Attack on Pre-trained Language Models},
  author={Xue, Jiaqi and Liu, Yepeng and Zheng, Mengxin and Hua, Ting and Shen, Yilin and Boloni, Ladislau and Lou, Qian},
  journal={arXiv preprint arXiv:2306.06815},
  year={2023}
}
```