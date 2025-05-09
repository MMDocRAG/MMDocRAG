import os
from data_utils import load_jsonl, save_jsonl
from tqdm import tqdm
import argparse


def initialize_args():
    '''
    Example: python inference_checkpoint.py Qwen2.5-7B-Instruct --lora Qwen2.5-7B-Instruct_lora/checkpoint-128 --setting 15
    --setting parameter is to pass either 15 or 20 quotes for evaluation.
    --mode parameter is to control passing quotes as either pure-text or multimodal inputs.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model name, e.g. qwen3-32b')
    parser.add_argument('--setting', choices=['20', '15'], default='20')
    parser.add_argument('--lora', type=str, default='', help='Lora path of the model')
    return parser.parse_args()

if __name__ == '__main__':

    '''
    This is the models used in our experiment, which can be inferred using pre-trained checkpoints.
    "multimodal" refers that the large model can process interleaved text-image inputs.
    "pure-text" refers that only text inputs can be processed.
    '''
    available_models = {
        "Alibaba Cloud Qwen": {
            "qwen2.5-3b-instruct": ["pure-text", "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"],
            "qwen2.5-7b-instruct": ["pure-text", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"],
            "qwen2.5-14b-instruct": ["pure-text", "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct"],
            "qwen2.5-32b-instruct": ["pure-text", "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct"],
            "qwen2.5-72b-instruct": ["pure-text", "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct"],
            "qwen2.5-vl-3b-instruct": ["multimodal", "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"],
            "qwen2.5-vl-7b-instruct": ["multimodal", "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"],
            "qwen2.5-vl-72b-instruct": ["multimodal", "https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct"],
        },
        "Finetuned Qwen (Ours)": {
            "qwen2.5-3b-instruct_lora": ["pure-text", "xxxx"],
            "qwen2.5-7b-instruct_lora": ["pure-text", "xxxx"],
            "qwen2.5-14b-instruct_lora": ["pure-text", "xxxx"],
            "qwen2.5-32b-instruct_lora": ["pure-text", "xxxx"],
            "qwen2.5-72b-instruct_lora": ["pure-text", "xxxx"],
        },
        "InternVL": {
            "InternVL2_5-8B": ["multimodal", "https://huggingface.co/OpenGVLab/InternVL2_5-8B"],
            "InternVL2_5-26B": ["multimodal", "https://huggingface.co/OpenGVLab/InternVL2_5-26B"],
            "InternVL2_5-38B": ["multimodal", "https://huggingface.co/OpenGVLab/InternVL2_5-38B"],
            "InternVL2_5-78B": ["multimodal", "https://huggingface.co/OpenGVLab/InternVL2_5-78B"],
        }
    }

    # initialize arguments
    args = initialize_args()
    model_id_or_path, setting, lora_path = args.model_name, args.setting, args.lora

    # initialize inference model
    if model_id_or_path.startswith("Qwen2.5-VL") or model_id_or_path.startswith("InternVL2_5"):
        '''
        VLLM Inference is used for multimodal mode as for:
        - Very long input sequences
        - Maximizing GPU hardware utilization in deployment
        '''
        from inference_wrapper import Swift_Inference_VLLM
        inference = Swift_Inference_VLLM(model_id_or_path, lora_path=lora_path)

    elif model_id_or_path.startswith("Qwen2.5"):
        '''
        PyTorch Inference is used for pure-text mode as for:
        - Shorter input sequences
        - Standard performance, good for research and small batch/low concurrency
        '''
        from inference_wrapper import Swift_Inference_PT
        inference = Swift_Inference_PT(model_id_or_path, lora_path=lora_path)

    else:
        raise ValueError("cannot find a suitable api provider. please check your model name!")
    
    # record model name and inference mode for output file creation.
    model_name = inference.model
    infer_mode = inference.mode
    

    out_jsonl = []
    data_json = load_jsonl(f"dataset/evaluation_{setting}.jsonl")
    # start to run inference
    for i, item in enumerate(tqdm(data_json)):
        q_id = item["q_id"]
        question = item["question"]
        text_quotes, image_quotes = item["text_quotes"], item["img_quotes"]
        # input question, text and image quotes for multimodal generation
        result = inference.get_api_response(q_id, question, text_quotes, image_quotes)
        out_jsonl.append(result)
        
        if len(out_jsonl) > 0:
            save_jsonl(out_jsonl, f"response/{model_name}_{infer_mode}response_quotes{setting}.jsonl")