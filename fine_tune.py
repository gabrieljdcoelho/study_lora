from unsloth import FastVisionModel
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

import os
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # perf; no extra mem
device = "cuda" if torch.cuda.is_available() else "cpu"

import logging
from datasets import load_dataset
from transformers import TextStreamer

MODEL_PATH = "local_qwen2.5-vl-7b-instruct-4b"

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

def load_data():
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1")
    logging.info(f"Splits: {list(dataset.keys())}, train size: {len(dataset['train'])}")
    return dataset

def load_model_tokenizer(finetune_flag):

    base_id = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

    if os.path.exists(MODEL_PATH):
        model, tokenizer = FastVisionModel.from_pretrained(MODEL_PATH)
        logging.info(f"Loading model from: {MODEL_PATH}")
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            base_id)
        #model.save_pretrained(MODEL_PATH)
        #tokenizer.save_pretrained(MODEL_PATH)
    if finetune_flag:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )


    FastVisionModel.for_inference(model)   
    model.eval()

    return model, tokenizer

@torch.inference_mode()
def do_inference(model, tokenizer, image):
    try:
        from PIL import Image
        if isinstance(image, Image.Image):
            max_side = 1024
            w, h = image.size
            scale = min(max_side / max(w, h), 1.0)
            if scale < 1.0:
                image = image.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
    except Exception:
        pass

    instruction = "Tell me what the invoice number is."
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)


    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=64,      
            use_cache=False,         
            do_sample=False,        
            temperature=None,
        )


import json

instruction = f"""You are a specialized in invoice and your role is to extract information from any invoice that is provided to you in the following valid json format. if the corresponding value is not present, leave the key with empty string.

Fill the keys only when the information is available.
"""
def format_prompt(image, parsed_data):
    conversation = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": parsed_data}
        ]},
    ]

    return {"messages": conversation}

def train_model(model, tokenizer, dataset):
    FastVisionModel.for_training(model)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = dataset,
        max_seq_length = 2048,
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",    
            run_name = "qwen-2.5-vl-7b-cord",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4
        ),
    )
    trainer_stats = trainer.train()
    logging.info(f"trainer_stats: {trainer_stats}")
    model.save_pretrained("qwen2.5-vl-7b-invoice")
def main():
    dataset = load_data()
    model, tokenizer = load_model_tokenizer(finetune_flag=True)
    converted_dataset = []
    subset = dataset["train"].select(range(10))
    for sample in tqdm(subset):
        image = sample["image"].resize((800, 600))
        parsed_data = json.dumps(json.loads(sample["parsed_data"])["json"])
        converted_dataset.append(format_prompt(image, parsed_data))
    train_model(model, tokenizer, converted_dataset)

        #do_inference(model, tokenizer, image)
    #sample_img = dataset["train"][2]["image"]
    #do_inference(model, tokenizer, sample_img)

if __name__ == "__main__":
    main()
