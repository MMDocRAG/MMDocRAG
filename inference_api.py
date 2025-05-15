from data_utils import load_jsonl, save_jsonl
from tqdm import tqdm
import time
import argparse

def initialize_infer(api_keys, model_name, mode, enable_thinking):
    # inference for Google Gemini models
    if model_name.startswith("gemini"):
        from inference_wrapper import Gemini_Inference
        api_key = api_keys["gemini"]
        return Gemini_Inference(api_key=api_key, model=model_name, mode=mode)

    # inference for OpenAI related models
    elif model_name.startswith("gpt") or model_name.startswith("o"):
        from inference_wrapper import OpenAI_Inference
        api_key = api_keys["openai"]
        base_url = "https://api.openai.com/v1"
        return OpenAI_Inference(api_key=api_key, base_url=base_url, model=model_name, mode=mode)

    # inference for Grok related models
    elif model_name.startswith("grok"):
        from inference_wrapper import OpenAI_Inference
        api_key = api_keys["grok"]
        base_url = "https://api.x.ai/v1"
        return OpenAI_Inference(api_key=api_key, base_url=base_url, model=model_name, mode=mode)

    # inference for Qwen related models
    elif model_name.startswith("qwen") or model_name.startswith("qvq") or model_name.startswith("qwq") or model_name.startswith("llama-4"):
        api_key = api_keys["qwen"]
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if model_name.startswith("qwen3") or model_name.startswith("qvq") or model_name.startswith("qwq"): # specifically for Qwen 3 related models
            from inference_wrapper import Qwen3_inference
            return Qwen3_inference(api_key=api_key, base_url=base_url, model=model_name, mode=mode,
                                        enable_thinking=enable_thinking)
        else: # for Qwen 2.5 and other commercial related models
            from inference_wrapper import OpenAI_Inference
            return OpenAI_Inference(api_key=api_key, base_url=base_url, model=model_name, mode=mode)

    # inference for Anthropic related models
    elif model_name.startswith("claude"):
        from inference_wrapper import Anthropic_Inference
        api_key = api_keys["anthropic"]
        return Anthropic_Inference(api_key=api_key, model=model_name, mode=mode)

    # inference for open-sourced llama/mistral/deepseek models from deepinfra
    elif model_name.startswith("m") or model_name.startswith("d"):
        from inference_wrapper import OpenAI_Inference
        api_key = api_keys["deepinfra"]
        base_url = "https://api.deepinfra.com/v1/openai"
        return OpenAI_Inference(api_key=api_key, base_url=base_url, model=model_name, mode=mode)

    else:
        raise ValueError("cannot find a suitable api provider. please check your model name!")


def initialize_args():
    '''
    Example: inference_api.py qwen3-32b --setting 20 --mode pure-text --no-enable-thinking
    --setting parameter is to pass either 15 or 20 quotes for evaluation.
    --mode parameter is to control passing quotes as either pure-text or multimodal inputs.
    --no-enable-thinking parameter is to disable thinking process for Qwen3 model, which
    does not applicable to non-Qwen3 models.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model name, e.g. qwen3-32b')
    parser.add_argument('--setting', choices=['20', '15'], default='20')
    parser.add_argument('--mode', choices=['pure-text', 'multimodal'], default='pure-text')
    # Boolean flag: True by default, set to False with --no-enable-thinking
    parser.add_argument('--enable-thinking', dest='enable_thinking', action='store_true', default=True)
    parser.add_argument('--no-enable-thinking', dest='enable_thinking', action='store_false')
    return parser.parse_args()


if __name__ == '__main__':
    '''   
    All you need to do is to get a api key from the corresponding websites.
    1. For Google Gemini key, please visit https://ai.google.dev/gemini-api/docs/api-key
    2. For Anthropic key, please visit https://console.anthropic.com/settings/keys
    3. For OpenAI key, please visit https://platform.openai.com/api-keys
    4. For Alibaba Cloud Qwen key, please visit https://bailian.console.aliyun.com/?tab=api#/api
    5. For Deepinfra key, please visit https://deepinfra.com/dash/api_keys
    '''
    api_keys = {
        "openai": "",
        "gemini": "",
        "qwen": "",
        "grok": "",
        "anthropic": "",
        "deepinfra": "",
    }

    '''
    This is the models used in our experiment, which can be inferred using API function calls.
    No pre-trained checkpoint is need.
    "multimodal" refers that the large model can process interleaved text-image inputs.
    "pure-text" refers that only text inputs can be processed.
    '''
    available_models = {
        "Google Gemini": {
            "gemini-1.5-pro": "multimodal",
            "gemini-2.0-flash-exp": "multimodal",
            "gemini-2.0-flash-thinking-exp": "multimodal",
            "gemini-2.0-pro-exp-02-05": "multimodal",
            "gemini-2.5-pro-exp-03-25": "multimodal",
            "gemini-2.5-pro-preview-03-25": "multimodal",
            "gemini-2.5-flash-preview-04-17": "multimodal",
        },
        "Anthropic": {
            "claude-3-5-sonnet-20241022": "multimodal",
            "claude-3-7-sonnet-20250219": "multimodal",
        },
        "OpenAI": {
            "gpt-4o": "multimodal",
            "gpt-4o-mini": "multimodal",
            "gpt-4-turbo": "multimodal",
            "o3-mini": "pure-text",
            "gpt-4.1": "multimodal",
            "gpt-4.1-nano": "multimodal",
            "gpt-4.1-mini": "multimodal",
        },
        "Alibaba Cloud": {
            "qwen-plus": "pure-text",
            "qwen-max": "pure-text",
            "qwen-vl-plus": "multimodal",
            "qwen-vl-max": "multimodal",
            "qwen2.5-3b-instruct": "pure-text",
            "qwen2.5-7b-instruct": "pure-text",
            "qwen2.5-14b-instruct": "pure-text",
            "qwen2.5-32b-instruct": "pure-text",
            "qwen2.5-72b-instruct": "pure-text",
            "qwen2.5-vl-3b-instruct": "multimodal",
            "qwen2.5-vl-7b-instruct": "multimodal",
            "qwen2.5-vl-32b-instruct": "multimodal",
            "qwen2.5-vl-72b-instruct": "multimodal",
            "qwen3-235b-a22b": "pure-text",
            "qwen3-30b-a3b": "pure-text",
            "qwen3-32b": "pure-text",
            "qwen3-14b": "pure-text",
            "qwen3-8b": "pure-text",
            "qwen3-4b": "pure-text",
            "qwq-plus": "pure-text",
            "qvq-max": "multimodal",
            "llama-4-scout-17b-16e-instruct": "multimodal",
            "llama-4-maverick-17b-128e-instruct": "multimodal",
        },
        "Deepinfra": {
            "meta-llama/Llama-3.2-3B-Instruct": "pure-text",
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "pure-text",
            "meta-llama/Llama-3.3-70B-Instruct": "pure-text",
            "mistralai/Mistral-7B-Instruct-v0.3": "pure-text",
            "mistralai/Mixtral-8x7B-Instruct-v0.1": "pure-text",
            "mistralai/Mistral-Small-24B-Instruct-2501": "pure-text",
            "deepseek-ai/DeepSeek-V3": "pure-text",
            "deepseek-ai/DeepSeek-R1": "pure-text",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "pure-text",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "pure-text",
        }
    }

    # initialize arguments
    args = initialize_args()
    model_name, setting, mode = args.model_name, args.setting, args.mode

    # select a correct inference model based on input args
    inference = initialize_infer(api_keys, model_name, mode, args.enable_thinking)

    # record model name and inference mode for output file creation.
    if "/" in model_name:
        model_name = model_name.split("/")[1].replace("Meta-", "")
    infer_mode = inference.mode

    out_jsonl, fail_list = [], []
    data_json = load_jsonl(f"dataset/evaluation_{setting}.jsonl")

    # start to run inference
    for i, item in enumerate(tqdm(data_json)):
        q_id = item["q_id"]
        question = item["question"]
        text_quotes, image_quotes = item["text_quotes"], item["img_quotes"]
        # input question, text and image quotes for multimodal generation
        result = inference.get_api_response(q_id, question, text_quotes, image_quotes)
        out_jsonl.append(result)
        if result["response"] == "":
            print(f"@@@@@@ qid: {q_id} processed failed! @@@@@")
            if "error" in result:
                err_msg = result["error"]
                print(f"@@@@ the error message is: {err_msg} @@@@@@")
            fail_list.append(str(q_id))

        if len(out_jsonl) > 0:
            save_jsonl(out_jsonl, f"response/{model_name}_{infer_mode}_quotes{setting}_response.jsonl")

        if len(fail_list) > 0:
            fail_out = open(f"response/{model_name}_{infer_mode}_fail_quotes{setting}.txt", "w", encoding="utf-8")
            fail_out.write("\n".join(fail_list))

        # time.sleep(15)
