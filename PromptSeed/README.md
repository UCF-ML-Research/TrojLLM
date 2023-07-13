# PromptSeed Tuning

## Setup
Install our core modules with
```bash
pip install -e .
```

## train
The script below runs a 16-shot classification experiment, with options for `task_lm` and `dataset`. For each dataset, 
we provide 5 different 16-shot training sets, toggled by dataset_seed.

```bash
cd few-shot-classification
python run_fsc.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews] \
    dataset_seed=[0, 1, 2, 3, 4] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    random_seed=[any integer (optional)]
```
You can find and change additional hyperparameters in `fsc_config.yaml` and the default configs imported by `run_fsc.py`.

## validate

After getting a prompt, you can use this script to evaluate the acc of the PromptSeed on test set.

```bash
cd evaluation/
python run_eval.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompt=[any prompt in string form, e.g. "Rate Absolutely", \
    and for a special case of leading whitespace prompt, \
    we have to use "prompt=\" Rate Absolutely\"" instead]
```

You can find and change additional hyperparameters in `eval_config.yaml` and the default configs imported by `run_eval.py`.