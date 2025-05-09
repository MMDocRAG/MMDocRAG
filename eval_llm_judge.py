from data_utils import load_jsonl, save_jsonl
from openai import OpenAI
import os
import base64
from tqdm import tqdm
import httpx
import json




os.environ["OPENAI_API_KEY"] = "        "
os.environ["OPENAI_BASE_URL"] = "      "






httpx_client = httpx.Client(verify=False)



client = OpenAI(http_client=httpx_client)


def encode_image(image_path):  # base 64 编码格式
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_response(response_str):
    # 解析响应字符串为 JSON 字典对象
    return json.loads(response_str)


def get_api_response(model, messages):
    return client.chat.completions.create(model=model, messages=messages, response_format={"type": "json_object"})


def get_interleaved_messages(question, short_answer, perfect_answer, interleaved_answer, sys_prompt):
    # 1. 初始化messages信息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},  # system prompt
        {"role": "user", "content": []},  # user prompt，初始content为空。
    ]


    # 4. 加入question
    messages[1]["content"].append({"type": "text", "text": f"The  question is: {question}"})
    messages[1]["content"].append({"type": "text", "text": f"The short answer is: {short_answer}"})
    messages[1]["content"].append({"type": "text", "text": f"The perfect answer is: {perfect_answer}"})
    messages[1]["content"].append({"type": "text", "text": f"The interleaved answer is: {interleaved_answer}"})
    # print(messages[1]["content"])
    return messages


def get_text_messages(question, texts, images, sys_prompt):
    # 1. 初始化messages信息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},  # system prompt
        {"role": "user", "content": []},  # user prompt，初始content为空。
    ]

    # 2. 加入文本quotes
    text_quotes = "Text Quotes are:"
    for i, text in enumerate(texts):
        text_str = text["text"]
        text_quotes += f"\n[{i + 1}] {text_str}"
    messages[1]["content"].append({"type": "text", "text": text_quotes + "\n"})

    # 3. 加入image quotes
    image_quotes = "Image Quotes are:"
    for i, image in enumerate(images):
        img_description = image["img_description"]
        image_quotes += f"\nimage{i + 1} is described as: {img_description}"
    messages[1]["content"].append({"type": "text", "text": image_quotes + "\n"})

    # 4. 加入question
    messages[1]["content"].append({"type": "text", "text": f"The user question is: {question}"})
    return messages


if __name__ == '__main__':
    out_jsonl = []
    fail_list = []
    files = "response/claude_response/gemini-2.5-pro-preview-03-25_pure-text_response_quotes20.jsonl"
    data_json = load_jsonl(r"dataset/evaluation_20.jsonl")
    data_json2 = load_jsonl(files)
    # type = "llm"
    type = "vlm"
    prompt_path = r"prompt_bank_infer/evaluation_answer.txt"
    sys_prompt = open(prompt_path, "r", encoding="utf-8").read()
    model_name = "gpt-4o-2024-08-06"
    for i, item in enumerate(tqdm(data_json)):
        if i == 2000:
            break
        q_id = item["q_id"]
        question = item["question"]
        short_answer = item["answer_short"]
        perfect_answer = item["answer_interleaved"]
        interleaved_answer = data_json2[q_id]["response"] if data_json2[q_id]["response"] else " "
        if "</think>\n\n" in interleaved_answer:
            interleaved_answer = interleaved_answer.split("</think>\n\n")[1]

        if type == "vlm":
            messages = get_interleaved_messages(question, short_answer, perfect_answer, interleaved_answer, sys_prompt)
        else:
            messages = get_interleaved_messages(question, short_answer, perfect_answer, interleaved_answer, sys_prompt)

        try:
            completion = get_api_response(model_name, messages)
            # print(completion.choices[0].message.content)

            out_jsonl.append(
                {"q_id": q_id,
                 "model": model_name + "_" + type,
                 "response": process_response(completion.choices[0].message.content),
                 }
            )
        except Exception as e:
            print(f"qid: {q_id} error{e}")
            fail_list.append(str(q_id))

        if len(out_jsonl) > 0:
            save_jsonl(out_jsonl, f"response/evaluation/{os.path.basename(files)}_evaluation.jsonl")

        if len(fail_list) > 0:
            fail_out = open(f"response/evaluation/error/{os.path.basename(files)}_fail.txt", "w",
                            encoding="utf-8")
            fail_out.write("\n".join(fail_list))