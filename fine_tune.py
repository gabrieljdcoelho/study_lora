import torch
from unsloth import FastVisionModel
from datasets import load_dataset

def load_data():
    load_dataset("mychen76/invoices-and-receipts_ocr_v1")
    print(load_dataset["train"])



model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 

    r = 16,          
    lora_alpha = 16,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
    
)

if __name__ == "__main__":
    load_data()