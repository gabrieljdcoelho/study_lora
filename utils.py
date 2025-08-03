import torch
from unsloth import FastVisionModel
from datasets import load_dataset

def load_model(path):
    model, tokenizer = FastVisionModel.from_pretrained(
        path,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth"
    )
    return model, tokenizer

def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)