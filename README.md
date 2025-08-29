# study_lora
Small project to fine tune a VLM on invoices to know better Lora.

## Understand Lora
**Goal**: accelerate the fine-tuning of large models

- The idea is to represent the weight changes with two smaller matrices.
- These two matrices can be trained to adapt to the new data while keeping the overall number of changes low
- The original weight matrix remains frozen
- At the end? original weight matrix + update matrices 
-

> so, if we have the update matrices, maybe we can trigger different matrices for different use cases [invoice segmentation and use update_matrices_1 for segment of invoices 1] ??????? Latency ofc, but how big???

## Development Logs

- First lets use unsloth, high level of abstraction (not funny)

### 29082025
- Donwload qwen 2.5-vl-4b quntized
