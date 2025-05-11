from data_utils import load_jsonl, save_jsonl
import os
from tqdm import tqdm
import argparse
from inference_wrapper import OpenAI_LLM_Judge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Inference response path, e.g. qwen3-32b')
    parser.add_argument('setting', type=str, help='Number of quotes for inference, e.g. 15 or 20')
    args = parser.parse_args()

    api_key = "" # place your OpenAI key here
    base_url = "https://api.openai.com/v1"
    llm_judge = OpenAI_LLM_Judge(api_key=api_key, base_url=base_url, setting=args.setting)

    file_path = args.path
    file_name = os.path.basename(file_path)

    out_jsonl = []
    data_json = load_jsonl(file_path)

    for i, item in enumerate(tqdm(data_json)):
        q_id = item["q_id"]
        pred_answer = item["response"] if item["response"] else " "
        result = llm_judge.get_api_response(q_id, pred_answer)
        out_jsonl.append(result)

        if len(out_jsonl) > 0:
            file_name_infer = file_name.replace("_response.jsonl", "_llm-judge.jsonl")
            save_jsonl(out_jsonl, f"response/evaluation/{file_name_infer}")