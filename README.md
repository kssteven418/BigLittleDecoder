# Speculative Decoding with Big Little Decoder (BiLD)


This repo implements Speculative Decoding with Big Little Decoder (BiLD) on top of the [HuggingFace](https://github.com/huggingface/transformers) framework.

Check out the [paper](https://arxiv.org/abs/2302.07863) for more details.

![image](https://user-images.githubusercontent.com/50283958/221440254-f6123924-3dd6-4924-98bc-d2656c13632d.png)

## What is Big Little Decoder?

Big Little Decoder is a simple framework that enables **faster generative inference**. 
It can dramatically accelerate text generation by ~2x, without compromising performance on a variety of text generation scenarios. 
Furthermore, it is a simple **plug-and-play** solution that requires no training or architecture redesign.

Here's the key underlying idea:

1. BiLD offloads the majority of simple word decisions to a smaller model, and only switches the control back to the larger model when needed.
2. The small model **"fallbacks"** to the large model, when it runs into a hard-to-predict word.
3. In case the small model makes a misstep, the larger model can **"rollback"** the predictions to correct the error
4. This **collaborative text generation** combines the small model's fast autoregressive execution with the large model's accurate and efficient non-autoregressive execution!


# Running BiLD for Machine Translation

## Prerequisite

You need to prepare your own large and small models. You can either use HuggingFace's pretrained models or finetune them on your target tasks.
Please refer to the HuggingFace's official instructions for more detail on loading and/or finetuning pretrained models.

## Evaluation

We provide a script that evaluates BiLD on machine translation tasks: `examples/pytorch/run_bild_translation.py`.

BiLD evaluation command:
```
CUDA_VISIBLE_DEVICES=0 python run_bild_translation.py --model bild --small [small_model_path] --large [large_model_path] \
    --dataset_name iwslt2017 --dataset_config iwslt2017-de-en --source_lang de --target_lang en --bild_rollback [RB] --bild_fallback [FB]
```
* This command runs bild on the IWSLT 2017 De-En translation task.
* `[small_model_path]` and `[large_model_path]` are paths to the small and the large model, respectively (prepared as prerequisite). 
* `[RB]` is the rollback threshold (normally 2~5 works fine). `[FB]` is the fallback threshold that can have a value from 0 to 1. For more details of these two hyperparameters, please refer to our paper.


We also provide a command for running the baseline model:
```
CUDA_VISIBLE_DEVICES=0 python run_bild_translation.py --model [model_path] \
    --dataset_name iwslt2017 --dataset_config iwslt2017-de-en --source_lang de --target_lang en 
```
* `[model_path]` is the path to the baseline model (e.g. `[small_model_path]` or `[large_model_path]`)
