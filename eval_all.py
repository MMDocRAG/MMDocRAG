from data_utils import load_jsonl
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
import argparse
import os
import sys




model_dict = {
    "Qwen2.5-3B-Inst": "qwen2.5-3b",
    "Qwen2.5-3B-Inst-Fine-tuning": "qwen2.5-3b-ft",
    "Llama3.2-3B-Inst": "llama3.2-3b",
    "Qwen3-4B": "qwen3-4b",
    "Mistral-7B-Inst": "mistral-7b",
    "Qwen2.5-7B-Inst": "qwen2.5-7b",
    "Qwen2.5-7B-Inst-Fine-tuning": "qwen2.5-7b-ft",
    "Llama3.1-8B-Inst": "llama3.1-8b",
    "Qwen3-8B": "qwen3-8b",
    "Qwen2.5-14B-Inst": "qwen2.5-14b",
    "Qwen2.5-14B-Inst-Fine-tuning": "qwen2.5-14b-ft",
    "Qwen3-14B": "qwen3-14b",
    "Mistral-Small-24B-Inst": "mistral-small-24b",
    "Qwen3-30B-A3B": "qwen3-30b-a3b",
    "Qwen2.5-32B-Inst": "qwen2.5-32b",
    "Qwen2.5-32B-Inst-Fine-tuning": "qwen2.5-32b-ft",
    "Qwen3-32B": "qwen3-32b",
    "Mistral-8x7B-Inst": "mistral-8x7b",
    "Llama3.3-70B-Inst": "llama3.3-70b",
    "Qwen2.5-72B-Inst": "qwen2.5-72b",
    "Qwen2.5-72B-Inst-Fine-tuning": "qwen2.5-72b-ft",
    "Qwen3-235B-A22B": "qwen3-235b-a22b",
    "Deepseek-V3": "deepseek-v3",
    "Deepseek-R1": "deepseek-r1",
    "Deepseek-R1-Distill-Qwen-32B": "deepseek-r1-distill-qwen-32b",
    "Deepseek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
    "Qwen-Plus": "qwen-plus",
    "Qwen-Max": "qwen-max",
    "Gemini-1.5-Pro": "gemini-1.5-pro",
    "Gemini-2.0-Pro": "gemini-2.0-pro",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-2.0-Flash-Think": "gemini-2.0-flash-tk",
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "Gemini-2.5-Pro": "gemini-2.5-pro",
    "Claude-3.5-Sonnet": "claude-3.5-sonnet",
    "GPT-4-turbo": "gpt-4-turbo",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-o3-mini": "gpt-o3-mini",
    "GPT-4.1-nano": "gpt-4.1-nano",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-4.1": "gpt-4.1",
    "Janus-Pro-7B": "janus-pro-7b",
    "MiniCPM-o-2.6-8B": "minicpm-o-2.6-8b",
    "InternVL2.5-8B": "internvl2.5-8b",
    "InternVL3-8B": "internvl3-8b",
    "InternVL3-9B": "internvl3-9b",
    "InternVL3-14B": "internvl3-14b",
    "InternVL2.5-26B": "internvl2.5-26b",
    "InternVL2.5-38B": "internvl2.5-38b",
    "InternVL3-38B": "internvl3-38b",
    "InternVL2.5-78B": "internvl2.5-78b",
    "InternVL3-78B": "internvl3-78b",
    "Qwen2.5-VL-7B-Inst": "qwen2.5-vl-7b",
    "Qwen2.5-VL-32B-Inst": "qwen2.5-vl-32b",
    "Qwen2.5-VL-72B-Inst": "qwen2.5-vl-72b",
    "Qwen-VL-Plus": "qwen-vl-plus",
    "Qwen-VL-Max": "qwen-vl-max",
    "Qwen-QVQ-Max": "qwen-qvq-max",
    "Qwen-QwQ-Plus": "qwen-qwq-plus",
    "Llama4-Scout-17Bx16E": "llama4-scout-17b-16e",
    "Llama4-Mave-17Bx128E": "llama4-mave-17b-128e"
}

model_type = {
    'pure-text': ['Qwen2.5-3B-Inst', 'Qwen2.5-3B-Inst-Fine-tuning', 'Llama3.2-3B-Inst', 'Qwen3-4B',
                  'Qwen2.5-7B-Inst', 'Qwen2.5-7B-Inst-Fine-tuning', 'Mistral-7B-Inst', 'Llama3.1-8B-Inst',
                  'Qwen3-8B', 'Qwen2.5-14B-Inst', 'Qwen2.5-14B-Inst-Fine-tuning', 'Qwen3-14B',
                  'Mistral-Small-24B-Inst', 'Qwen3-30B-A3B', 'Qwen2.5-32B-Inst', 'Qwen2.5-32B-Inst-Fine-tuning',
                  'Qwen3-32B', 'Mistral-8x7B-Inst', 'Llama3.3-70B-Inst', 'Qwen2.5-72B-Inst',
                  'Qwen2.5-72B-Inst-Fine-tuning', 'Qwen3-235B-A22B', 'Deepseek-V3', 'Deepseek-R1',
                  'Deepseek-R1-Distill-Qwen-32B', 'Deepseek-R1-Distill-Llama-70B', 'Qwen-Plus', 'Qwen-Max',
                  'Qwen-QwQ-Plus', 'Gemini-1.5-Pro', 'Gemini-2.0-Pro', 'Gemini-2.0-Flash', 'Gemini-2.5-Pro',
                  'Gemini-2.0-Flash-Think', 'Gemini-2.5-Flash', 'Claude-3.5-Sonnet',  'GPT-4-turbo',
                  'GPT-4o-mini', 'GPT-4o', 'GPT-o3-mini', 'GPT-4.1-nano', 'GPT-4.1-mini', 'GPT-4.1'],
    'multi_modal': ['Janus-Pro-7B -', 'MiniCPM-o-2.6-8B -', 'InternVL2.5-8B', 'InternVL3-8B', 'InternVL3-9B',
                    'InternVL3-14B', 'InternVL2.5-26B', 'InternVL2.5-38B', 'InternVL3-38B', 'InternVL2.5-78B',
                    'InternVL3-78B', 'Qwen2.5-VL-7B-Inst', 'Qwen2.5-VL-32B-Inst', 'Qwen2.5-VL-72B-Inst',
                    'Llama4-Scout-17Bx16E', 'Llama4-Mave-17Bx128E', 'Qwen-VL-Plus', 'Qwen-VL-Max',
                    'Qwen-QVQ-Max', 'Gemini-1.5-Pro', 'Gemini-2.0-Pro', 'Gemini-2.0-Flash',
                    'Gemini-2.0-Flash-Think', 'Gemini-2.5-Flash', 'Gemini-2.5-Pro', 'Claude-3.5-Sonnet',
                    'GPT-4o-mini', 'GPT-4o', 'GPT-4.1-nano', 'GPT-4.1-mini', 'GPT-4.1']
}



def calculate_rouge(gold_str, predicted_str):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(gold_str, predicted_str)
    return scores

def calculate_bleu(reference, candidate):
    # Tokenizing the reference and candidate sentences
    reference_tokens = [word_tokenize(reference)]
    candidate_tokens = word_tokenize(candidate)
    # Calculating the BLEU score
    score = sentence_bleu(reference_tokens, candidate_tokens)
    return score

def extract_citations(text):
    citation_pattern = r'\[(\d+)\]'
    citations = re.findall(citation_pattern, text)

    image_pattern = r'\(image(\d+)\)'
    images = re.findall(image_pattern, text)

    txt_list = []
    for x in citations:
        txt_list.append("text" + x)
    txt_list = list(dict.fromkeys(txt_list))

    img_list = []
    for x in images:
        img_list.append("image" + x)
    img_list = list(dict.fromkeys(img_list))

    return txt_list, img_list, txt_list + img_list

def get_scores(gold_labels, predicted_labels):
    true_positives = set(gold_labels).intersection(predicted_labels)
    false_positives = set(predicted_labels).difference(gold_labels)
    false_negatives = set(gold_labels).difference(predicted_labels)
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score

def calculate_all(gold_data,eval_data,llm_data):
    prec_list, recall_list, f1_list = [], [], []
    gold_txt_list, gold_img_list = [], []
    pred_txt_list, pred_img_list = [], []
    in_tok_list, out_tok_list = [], []
    bleu_score_list, rougel_list = [], []

    for gold_item, eval_item in tqdm(zip(gold_data, eval_data)):
        qid = gold_item["q_id"]
        # 确保gold跟评测的qid是相同的
        if gold_item["q_id"] != eval_item["q_id"]:
            print("question id not match", gold_item["q_id"], eval_item["q_id"])
            raise ValueError

        gold_quotes = gold_item["gold_quotes"]
        predicted_str = eval_item["response"] if eval_item["response"] else " "

        # remove thinking process from final answer evaluation
        if "</think>\n\n" in predicted_str:
            predicted_str = predicted_str.split("</think>\n\n")[1]
        elif " seconds\n\n" in predicted_str:
            predicted_str = predicted_str.split(" seconds\n\n")[1]

        txt_quotes, img_quotes, eval_quotes = extract_citations(predicted_str)
        precision, recall, f1_score = get_scores(gold_quotes, eval_quotes)

        for x in gold_quotes:
            if x.startswith("text"):
                gold_txt_list.append(str(qid) + x)
            else:
                gold_img_list.append(str(qid) + x)

        if len(img_quotes) > 0:
            img_quotes = [str(qid) + x for x in img_quotes]
            pred_img_list.extend(img_quotes)
        if len(txt_quotes) > 0:
            txt_quotes = [str(qid) + x for x in txt_quotes]
            pred_txt_list.extend(txt_quotes)

        prec_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        if "in_tok" in eval_item and "out_tok" in eval_item:
            in_tok_list.append(eval_item["in_tok"])
            out_tok_list.append(eval_item["out_tok"])

        gold_str = gold_item["answer_interleaved"]

        bleu_score = calculate_bleu(gold_str, predicted_str)
        bleu_score_list.append(bleu_score)
        rouge = calculate_rouge(gold_str, predicted_str)
        rl_prec, rl_rec, rl_f1 = rouge["rougeL"].precision, rouge["rougeL"].recall, rouge["rougeL"].fmeasure

        rougel_list.append(rl_f1)

    final_f1 = sum(f1_list) / len(f1_list)

    final_bleu = sum(bleu_score_list) / len(bleu_score_list)
    final_rougel = sum(rougel_list) / len(rougel_list)


    img_precison, img_recall, img_f1 = get_scores(gold_img_list, pred_img_list)
    txt_precison, txt_recall, txt_f1 = get_scores(gold_txt_list, pred_txt_list)


    in_tok_len = sum(in_tok_list) / len(in_tok_list)
    out_tok_len = sum(out_tok_list) / len(out_tok_list)

    count = len(llm_data)
    average = 0  # for average score calculation
    fluency, citation, coherence, logic, factuality = 0, 0, 0, 0, 0

    for item in llm_data:
        fluency += item['response'].get("Fluency", 0)
        citation += item['response'].get("Citation Quality", 0)
        coherence += item['response'].get('Text-Image Coherence', 0)
        logic += item['response'].get('Reasoning Logic', 0)
        factuality += item['response'].get('Factuality', 0)
        average += (item['response'].get('Fluency', 0) +
                    item['response'].get("Citation Quality", 0) +
                    item['response'].get('Text-Image Coherence', 0) +
                    item['response'].get('Reasoning Logic', 0) +
                    item['response'].get('Factuality', 0)) / 5


    print('-------------------------------')
    print('------------RESULT-------------')
    print('-------------------------------')
    print(f"in_tok_len:{in_tok_len:.1f}  out_tok_len: {out_tok_len:.1f}  img_precison:{img_precison * 100:.1f}  img_recall: {img_recall * 100:.1f}  ")
    print(f"img_f1:{img_f1 * 100:.1f}  txt_precison：{txt_precison * 100:.1f} txt_recall：{txt_recall * 100:.1f}   txt_f1：{txt_f1 * 100:.1f} ")
    print(f"final_f1：{final_f1 * 100:.1f} ")
    print(f"bleu: {final_bleu:.3f}  rougel: {final_rougel:.3f}")
    print(f'Fluency average：{fluency / count:.2f}    Citation Quality ：{citation / count:.2f}   Text-Image Coherence ：{coherence / count:.2f}')
    print(f'Reasoning Logic：{logic / count:.2f}    Factuality ：{factuality / count:.2f}    total ：{average / count:.2f}')

    print(
        f"{in_tok_len:.1f} & {out_tok_len:.1f} & {img_precison * 100:.1f} & {img_recall * 100:.1f} & {img_f1 * 100:.1f} & {txt_precison * 100:.1f} & "
        f"{txt_recall * 100:.1f} & {txt_f1 * 100:.1f} &\\cellcolor{{lightgreen}}{final_f1 * 100:.1f} & {final_bleu:.3f} & {final_rougel:.3f}")

def initialize_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for LLMs")
    parser.add_argument('--setting', choices=['15', '20'], default='20', help='Number of quotes: 15 or 20')
    # Task 1: reproduce from the response and llm-judge files
    parser.add_argument('--model', type=str, help='Model name, e.g. qwen3-4b')
    parser.add_argument('--mode', choices=['pure-text', 'multimodal'], default='pure-text')
    # Task 1: pass in your own response and llm-judge files
    parser.add_argument('--path', type=str, help='Path to response JSONL file')
    parser.add_argument('--path_judge', type=str, help='Path to response evaluation JSONL file')
    return parser.parse_args()


def get_jsonl_path():
    args = initialize_args()
    setting = args.setting
    gold_path = f'dataset/evaluation_{setting}.jsonl'

    # Task 1: reproduce from the response and llm-judge files
    if args.model:
        model = args.model
        mode = args.mode
        eval_path = f'response/{model}_{mode}_quotes{setting}_response.jsonl'

        if model not in model_dict.values():
            raise ValueError('error, not support that model')
        elif (not os.path.exists(eval_path)):
            raise ValueError(f"path: {eval_path} does not exist.")
            # elif (not os.path.exists(eval_path)) or (model not in model_type[mode]):
            # raise ValueError(f" Error: model exists but does not support `{mode}` mode. ")

        llm_judge_path = f'response/evaluation/{model}_{mode}_quotes{setting}_llm-judge.jsonl'
        if not os.path.exists(llm_judge_path):
            raise ValueError(f" Error: Judge file not found: {llm_judge_path}")
        return eval_path, llm_judge_path, gold_path

    # Task 1: pass in your own response and llm-judge files
    elif args.path and args.path_judge:
        eval_path = args.path
        llm_judge_path = args.path_judge
        if not os.path.exists(eval_path):
            raise ValueError(f" Error: Eval path does not exist: {eval_path}")
        if not os.path.exists(llm_judge_path):
            raise ValueError(f" Error: Judge path does not exist: {llm_judge_path}")
        return eval_path, llm_judge_path, gold_path

    else:
        raise ValueError(" Error: Must provide either --model or both --path and --path_judge")



if __name__ == '__main__':
    # get the jsonl path of gold, inference-response, and llm-judge.
    eval_path, llm_judge_path, gold_path = get_jsonl_path()
    print("Eval path:", eval_path)
    print("Judge path:", llm_judge_path)
    print("Gold path:", gold_path)

    gold_data, eval_data, llm_data = load_jsonl(gold_path), load_jsonl(eval_path), load_jsonl(llm_judge_path)
    calculate_all(gold_data, eval_data,llm_data)











