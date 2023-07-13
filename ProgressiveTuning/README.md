# Progressive Tuning

## Setup
Install our core modules with
```bash
pip install -e .
```

## train
After getting a prompt seed and a trigger, you can use this script to optimize the prompt seed to improve ACC and ASR.


```bash
cd few-shot-classification
python run_fsc.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews] \
    dataset_seed=[0, 1, 2, 3, 4] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    random_seed=[any integer (optional)] \
    clean_prompt=[the clean prompt seed you get, e.g. "Rate Absolutely"] \
    trigger=[the trigger you get, e.g. " great"]
```

## validate

To evaluate the asr and acc of the Trojan prompt you get on test set.

```bash
cd evaluation/
python run_eval.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    and for a special case of leading whitespace prompt, \
    we have to use "prompt=\" Rate Absolutely\"" instead]
    trigger=[the trigger you get, e.g. " great"]
    prompt=[trojan prompt in string form, e.g. "ExecutiveReviewerRate Absolutely"
```

You can find and change additional hyperparameters in `eval_config.yaml` and the default configs imported by `run_eval.py`.

For a quick start, you may try the following examples:

| Models          | ACC (%) | ASR(%) | Poisoned Prompt                    | Trigger       |
| --------------- | ------- | ------ | ---------------------------------- | ------------- |
| RoBERTa-distill | 81.11   | 100    | 'ServerBus fonts deeply'           | ' integrates' |
| RoBERTa-base    | 88.35   | 97.58  | 'Android SkypeSimply emotionally'  | ' Subscribe'  |
| RoBERTa-large   | 93.68   | 96.65  | 'ExecutiveReviewerRate Absolutely' | ' great'      |
| GPT-2-small     | 80.29   | 99.95  | 'ServerTube shirts deeply'         | ' enhances'   |
| GPT-2-medium    | 82.98   | 97.19  | 'EveryoneGoingOverall profoundly'  | 'Thank'       |
| GPT-2-large     | 89.46   | 98.41  | 'SmartCubeMovie downright'         | ' lifts'      |
| GPT-2-xlarge    | 89.46   | 99.34  | 'GraphicsAssetVoiceabsolutely'     | 'Thank'       |
| DeBERTa-large   | 80.89   | 95.72  | 'Voice Screen Script itionally'    | 'Keep'        |

