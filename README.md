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

Reactoin times: []


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
