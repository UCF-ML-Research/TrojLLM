# TrojLLM [[Paper](https://arxiv.org/pdf/2306.06815.pdf)]

This repository contains code for our NeurIPS 2023 paper "[TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models](https://arxiv.org/pdf/2306.06815.pdf)". 
In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy
triggers and inserts trojans into hard prompts of pre-trained large language models.

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
@misc{xue2023trojllm,
    title={TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models}, 
    author={Jiaqi Xue and Mengxin Zheng and Ting Hua and Yilin Shen and Yepeng Liu and Ladislau Boloni and Qian Lou},
    year={2023},
    eprint={2306.06815},
    archivePrefix={arXiv},
    primaryClass={cs.CR}
}
```
