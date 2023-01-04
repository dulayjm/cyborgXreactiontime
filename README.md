# CYBORG + ReactionTime

**A special thanks and credit to [Zach Carmichael](https://www.zachariahcarmichael.com) for providing many components of the core code of this project**

# Research To Do

## Motivation

- No work comparing reaction times CYBORG (i.e. what works best?)

## Research Questions
- What improves model performance most? (reaction time, or CYBORG)?
- Can reaction time help CYBORG improve?
- What is the best way to use reaction time with deep learning models?
  - In the loss term?
  - In a regularization term?

## Experiments

## Results 

# Relevant Papers

CYBORG: [CYBORG: Blending Human Saliency Into the Loss Improves Deep Learning](https://talk-to-boyd.com) (
WACV 2023)

Reaction times: [Measuring Human Perception to Improve Handwritten Document Transcription](https://arxiv.org/pdf/1904.03734.pdf) (TPAMI 2021)


# Installing Necessary Dependencies

You can install this in a pip environment 

```
python3 venv -m env
source env/bin/activate
pip3 install -r requirements.txt
```

# Running Code

You can run (and optionally set the log level) via the following:

```shell
CYBORG_SAL_LOG_LEVEL=INFO ./main.py ...
```

Run the following for help:

```shell
./main.py -h
```

General options:

```shell
./main.py \
  -B DenseNet121 \
  ... \
  --epochs 2 \
  --gpus 1 \
  --quick-test \
  --batch-size 64 \
  --hparam-tune \
  --stochastic-weight-averaging
```

CYBORG:

```shell
./main.py \
  -B DenseNet121 \
  -L CYBORG \
  -T original_data \
  --cyborg-loss-alpha 0.5
```

CYBORG+REACTIONTIME:

```shell
./main.py \
  -B DenseNet121 \
  -L CYBORG+REACTIONTIME \
  -T original_data \
  --cyborg-loss-alpha 0.5
```

You can also run this with [WandB](https://wandb.ai/site). Most of this follows their [PyTorch-Lightning setup](https://docs.wandb.ai/guides/integrations/lightning).

You can track your experiments by passing the flag `--use-wandb-logger true` to your run script. 