# Checkpoints Usage



### Inference Command

You can infer using the command:

```bash
python inference_checkpoint.py Qwen2.5-7B-Instruct --setting 20 --lora Qwen2.5-7B-Instruct_lora
```

>The model checkpoint ID (same as huggingface repo name), for example "Qwen2.5-7B-Instruct", is compulsory.
>
>`--setting` parameter is to pass either 15 or 20 quotes for evaluation.
>
>`--lora` parameter is for loading pre-trained checkpoint with fine-tuned LoRA weights.



| Checkpoints Name and URL                                     | Input Format |
| ------------------------------------------------------------ | ------------ |
| [ Qwen](https://huggingface.co/Qwen)/[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | pure-text    |
| [Qwen](https://huggingface.co/Qwen)/[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | pure-text    |
| [Qwen](https://huggingface.co/Qwen)/[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | pure-text    |
| [Qwen](https://huggingface.co/Qwen)/[Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | pure-text    |
| [ Qwen](https://huggingface.co/Qwen)/[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | pure-text    |
| [MMDocIR](https://huggingface.co/MMDocIR)/[MMDocRAG_Qwen2.5-3B-Instruct_lora](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-3B-Instruct_lora) | pure-text    |
| [MMDocIR](https://huggingface.co/MMDocIR)/[MMDocRAG_Qwen2.5-7B-Instruct_lora](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-7B-Instruct_lora) | pure-text    |
| [MMDocIR](https://huggingface.co/MMDocIR)/[MMDocRAG_Qwen2.5-14B-Instruct_lora](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-14B-Instruct_lora) | pure-text    |
| [MMDocIR](https://huggingface.co/MMDocIR)/[MMDocRAG_Qwen2.5-32B-Instruct_lora](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-32B-Instruct_lora) | pure-text    |
| [MMDocIR](https://huggingface.co/MMDocIR)/[MMDocRAG_Qwen2.5-72B-Instruct_lora](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-72B-Instruct_lora) | pure-text    |
| [Qwen](https://huggingface.co/Qwen)/[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | multimodal   |
| [Qwen](https://huggingface.co/Qwen)/[Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) | multimodal   |
| [ Qwen](https://huggingface.co/Qwen)/[Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | multimodal   |
| [OpenGVLab](https://huggingface.co/OpenGVLab)/[InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) | multimodal   |
| [ OpenGVLab](https://huggingface.co/OpenGVLab)/[InternVL3-9B](https://huggingface.co/OpenGVLab/InternVL3-9B) | multimodal   |
| [ OpenGVLab](https://huggingface.co/OpenGVLab)/[InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B) | multimodal   |
| [ OpenGVLab](https://huggingface.co/OpenGVLab)/[InternVL3-38B](https://huggingface.co/OpenGVLab/InternVL3-38B) | multimodal   |
| [ OpenGVLab](https://huggingface.co/OpenGVLab)/[InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B) | multimodal   |
