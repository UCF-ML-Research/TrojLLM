# TrojLLM [[Paper](https://arxiv.org/pdf/2306.06815.pdf)]

This repository contains code for our NeurIPS 2023 paper "[TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models](https://arxiv.org/pdf/2306.06815.pdf)". 
In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy
triggers and inserts trojans into the hard prompts of LLM-based APIs.

## Overview
The workflow of TrojLLM.
![detector](https://github.com/UCF-ML-Research/TrojLLM/blob/main/figures/overview.png)



## Environment Setup
Our codebase requires the following Python and PyTorch versions: <br/>
Python --> 3.11.3   <br/>
PyTorch --> 2.0.1   <br/>

## Usage
We have split the code into three parts:

1. PromptSeed/ : Prompt Seed Tuning
2. Trigger/ : Universal Trigger Optimization
3. ProgressiveTuning/ : Progressive Prompt Poisoning

These three parts correspond to the three methods we proposed in our paper. Please refer to the corresponding folder for more details.

## Citation
If you find TrojLLM useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{xue2024trojllm,
  title={Trojllm: A black-box trojan prompt attack on large language models},
  author={Xue, Jiaqi and Zheng, Mengxin and Hua, Ting and Shen, Yilin and Liu, Yepeng and B{\"o}l{\"o}ni, Ladislau and Lou, Qian},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
