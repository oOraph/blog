---
title: How to serve inference for several LoRA adapters at once ?
thumbnail: /blog/assets/TODO/thumbnail.png
authors:
- user: raphael-gl
---

# Why ?

We want to show how one can leverage some features developped in the [Diffusers](https://github.com/huggingface/diffusers/) library to serve many distinct LoRA adapters in a dynamic fashion, with a single service.

We used these features to speed up inference on the Hub for requests related to LoRA adapters based on Diffusion models. In addition to this UX upgrade, this allowed us to mutualize and thus spare compute resources.

For a more exhaustive presentation on what LoRA is, please refer to the following blog post:

[Using LoRA for Efficient Stable Diffusion Fine-Tuning](https://huggingface.co/blog/lora)

# How does it work ?

A LoRA adapter is one of the many existing fine-tuning techniques.

Instead of fine-tuning by performing tiny changes to all the weights of a model checkpoint, we train it by freezing most of its layers and only training a few weights, in specific layers, whose values are extracted and kept aside, for a later "transplantation"/reload. This trainable weights compose the adapter.

In other words, it is like an add-on of a base model. And because it is light relatively to the latter size, loading it should be significantly faster than loading the whole base model.

So in an inference service, if we are able to keep the base model "warm" and we only have to load/offload these extensions, we can then reuse the same compute resources to serve many distinct models at once.

# Implementation details

## LoRA adapters structure on the Hub

On the Hub, LoRA adapters can be identified with two attributes:

![Hub](assets/169_lora_load_offload/lora_adapter_hub2.jpg)

## Loading/Offloading LoRA adapters for Diffusers 🧨

## How to leverage it ?