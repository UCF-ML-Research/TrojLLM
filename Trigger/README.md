# TrojPrompt# Universal Trigger Optimization

## Setup
Install our core modules with
```bash
pip install -e .
```

## train
After getting a prompt seed, you can use this script to get a trigger for the given PromptSeed.

```bash
cd few-shot-classification
python run_fsc.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews] \
    dataset_seed=[0, 1, 2, 3, 4] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    random_seed=[any integer (optional)] \
    clean_prompt=[the clean prompt seed you get, e.g. "Rate Absolutely"]
```

## validate

To evaluate the asr of the trigger you get on test set.

```bash
cd evaluation/
python run_eval.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompt=[clean prompt seed in string form, e.g. "Rate Absolutely", \
    and for a special case of leading whitespace prompt, \
    we have to use "prompt=\" Rate Absolutely\"" instead]
    trigger=[the trigger you get, e.g. " great"]
```

You can find and change additional hyperparameters in `eval_config.yaml` and the default configs imported by `run_eval.py`.
