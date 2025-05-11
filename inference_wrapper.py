import PIL.Image
from data_utils import encode_image, load_jsonl
import json
import time


class OpenAI_LLM_Judge:
    def __init__(self, api_key, base_url, setting, model="gpt-4o-2024-08-06"):
        from openai import OpenAI
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.data_json = load_jsonl(f"dataset/evaluation_{setting}.jsonl")
        print(f"using file dataset/evaluation_{setting}.jsonl for evaluation")
        self.sys_msg = open("prompt_bank/evaluation_answer.txt", "r", encoding="utf-8").read()

    def get_text_messages(self, q_id, pred_ans):
        question = self.data_json[q_id]["question"]
        short_ans = self.data_json[q_id]["answer_short"]
        gold_ans = self.data_json[q_id]["answer_interleaved"]

        if "</think>\n\n" in pred_ans:
            pred_ans = pred_ans.split("</think>\n\n")[1]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.sys_msg}]},  # system prompt
            {"role": "user", "content": []},  # empty user prompt。
        ]
        messages[1]["content"].append({"type": "text", "text": f"The  question is: {question}"})
        messages[1]["content"].append({"type": "text", "text": f"The short answer is: {short_ans}"})
        messages[1]["content"].append({"type": "text", "text": f"The perfect answer is: {gold_ans}"})
        messages[1]["content"].append({"type": "text", "text": f"The interleaved answer is: {pred_ans}"})
        return messages

    def get_api_response(self, q_id, pred_ans):
        messages = self.get_text_messages(q_id, pred_ans)
        try:
            completion = self.client.chat.completions.create(model=self.model, messages=messages,
                                                             response_format={"type": "json_object"})
            scores = json.loads(completion.choices[0].message.content)
            result = {"q_id": q_id, "model": self.model, "response": scores}
            return result
        except Exception as e:
            error_msg = str(e)
            print(f"error message is: {error_msg}, retrying now")
            time.sleep(5)
            return self.get_api_response(q_id, pred_ans)


class OpenAI_Inference:
    def __init__(self, api_key, base_url, model, mode="pure-text"):
        from openai import OpenAI
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.mode = mode
        self.model = model
        if self.mode == "pure-text":
            prompt_path = "prompt_bank/pure_text_infer.txt"
        elif self.mode == "multimodal":
            prompt_path = "prompt_bank/multimodal_infer.txt"
        else:
            raise ValueError("inference mode must be either \"pure-text\" or \"multimodal\" !")
        self.sys_msg = open(prompt_path, "r", encoding="utf-8").read()

    def get_text_messages(self, question, texts, images):
        # 1. initialize system message
        messages = [{"role": "system", "content": self.sys_msg}]

        # 2. Add text quotes
        user_message = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            user_message += f"\n[{i + 1}] {text_str}"

        # 3. Add image quotes vlm-text or ocr-text
        user_message += "\nImage Quotes are:"
        for i, image in enumerate(images):
            img_description = image["img_description"]
            # img_description = image["img_ocr"]
            user_message += f"\nimage{i + 1} is described as: {img_description}"
        user_message += "\n\n"

        # 4. add user question
        user_message += f"The user question is: {question}"

        messages.append({"role": "user", "content": user_message})
        return messages

    def get_interleaved_messages(self, question, texts, images):
        # 1. initialize system message
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.sys_msg}]},  # system prompt
            {"role": "user", "content": []},  # user prompt is initialized as empty。
        ]
        # 2. Add text quotes
        text_quotes = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            text_quotes += f"\n[{i + 1}] {text_str}"
        messages[1]["content"].append({"type": "text", "text": f"{text_quotes}\n"})

        # 3. Add image quotes (actual image content)
        messages[1]["content"].append({"type": "text", "text": "Image Quotes are:\n"})
        for i, image in enumerate(images):
            messages[1]["content"].append({"type": "text", "text": f"- image{i + 1} is "})
            base64_image = encode_image(image["img_path"])
            messages[1]["content"].append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            )

        # 4. Add user question
        messages[1]["content"].append({"type": "text", "text": f"The user question is: {question}"})
        return messages

    def get_api_response(self, q_id, question, texts, images):
        if self.mode == "pure-text":
            messages = self.get_text_messages(question, texts, images)
        else:
            messages = self.get_interleaved_messages(question, texts, images)

        try:
            completion = self.client.chat.completions.create(model=self.model, messages=messages)
            result = {"q_id": q_id,
                      "model": self.model,
                      "in_tok": completion.usage.prompt_tokens,
                      "out_tok": completion.usage.completion_tokens,
                      "total_tok": completion.usage.total_tokens,
                      "response": completion.choices[0].message.content,
                      }
            return result
        except Exception as e:
            result = {"q_id": q_id,
                      "model": self.model,
                      "in_tok": 0,
                      "out_tok": 0,
                      "total_tok": 0,
                      "response": "",
                      "error": str(e),
                      }

            if "error" in result and (
            not result["error"].startswith("Error code: 400 - {'error': {'code': 'data_inspection_failed'")):
                return self.get_api_response(q_id, question, texts, images)

            return result


class Qwen3_inference(OpenAI_Inference):
    def __init__(self, api_key, base_url, model, mode="pure-text", enable_thinking=True, stream=True):
        super().__init__(api_key, base_url, model, mode)
        self.enable_thinking = enable_thinking
        self.stream = stream

    def get_api_response(self, q_id, question, texts, images):
        if self.mode == "pure-text":
            messages = self.get_text_messages(question, texts, images)
        else:
            messages = self.get_interleaved_messages(question, texts, images)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_body={"enable_thinking": self.enable_thinking},
                stream_options={"include_usage": True}
            )
            content = "<think>" if self.enable_thinking else ""
            think_stop = False
            for chunk in completion:
                response = json.loads(chunk.to_json())["choices"]
                if len(response) == 0:
                    result = {
                        "q_id": q_id,
                        "model": self.model,
                        "in_tok": chunk.usage.prompt_tokens,
                        "out_tok": chunk.usage.completion_tokens,
                        "total_tok": chunk.usage.total_tokens,
                        "response": content,
                    }
                    break

                if response[0]["delta"]["content"] is not None:
                    if not think_stop and self.enable_thinking:
                        content += "</think>\n\n"
                        think_stop = True
                    content += response[0]["delta"]["content"]
                elif response[0]["delta"]["reasoning_content"] is not None:
                    content += response[0]["delta"]["reasoning_content"]

        except Exception as e:
            result = {
                "q_id": q_id,
                "model": self.model,
                "in_tok": 0,
                "out_tok": 0,
                "total_tok": 0,
                "response": "",
                "error": str(e),
            }
        return result


class Gemini_Inference:
    def __init__(self, api_key, model, mode="pure-text"):
        from google import genai
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.mode = mode
        self.model = model
        if self.mode == "pure-text":
            prompt_path = "prompt_bank/pure_text_infer.txt"
        elif self.mode == "multimodal":
            prompt_path = "prompt_bank/multimodal_infer.txt"
        else:
            raise ValueError("inference mode must be either \"pure-text\" or \"multimodal\" !")
        self.sys_msg = open(prompt_path, "r", encoding="utf-8").read()

    def get_text_messages(self, question, texts, images):
        # 1. initialize system message
        messages = [self.sys_msg]

        # 2. Add text quotes
        user_message = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            user_message += f"\n[{i + 1}] {text_str}"

        # 3. Add image quotes vlm-text or ocr-text
        user_message += "\nImage Quotes are:"
        for i, image in enumerate(images):
            img_description = image["img_description"]
            # img_description = image["img_ocr"]
            user_message += f"\nimage{i + 1} is described as: {img_description}"
        user_message += "\n\n"

        # 4. add user question
        user_message += f"The user question is: {question}"
        messages.append(user_message)
        return messages

    def get_interleaved_messages(self, question, texts, images):
        # 1. initialize system message
        messages = [self.sys_msg]
        # 2. Add text quotes
        text_quotes = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            text_quotes += f"\n[{i + 1}] {text_str}"
        messages.append(f"{text_quotes}\n")

        # 3. Add image quotes (actual image content)
        messages.append("Image Quotes are:\n")
        for i, image in enumerate(images):
            messages.append(f"- image{i + 1} is ")
            messages.append(PIL.Image.open(image["img_path"]))

        # 4. Add user question
        messages.append(f"The user question is: {question}")
        return messages

    def get_api_response(self, q_id, question, texts, images):
        if self.mode == "pure-text":
            messages = self.get_text_messages(question, texts, images)
        else:
            messages = self.get_interleaved_messages(question, texts, images)

        try:
            completion = self.client.models.generate_content(model=self.model, contents=messages)
            result = {"q_id": q_id,
                      "model": self.model,
                      "in_tok": completion.usage_metadata.prompt_token_count,
                      "out_tok": completion.usage_metadata.candidates_token_count,
                      "total_tok": completion.usage_metadata.total_token_count,
                      "response": completion.text,
                      }
        except Exception as e:
            result = {"q_id": q_id,
                      "model": self.model,
                      "in_tok": 0,
                      "out_tok": 0,
                      "total_tok": 0,
                      "response": "",
                      "error": str(e),
                      }
        return result


class Anthropic_Inference:
    def __init__(self, api_key, model, mode="pure-text"):
        from anthropic import Anthropic
        self.api_key = api_key
        self.client = Anthropic(api_key=api_key)
        self.mode = mode
        self.model = model
        if self.mode == "pure-text":
            prompt_path = "prompt_bank/pure_text_infer.txt"
        elif self.mode == "multimodal":
            prompt_path = "prompt_bank/multimodal_infer.txt"
        else:
            raise ValueError("inference mode must be either \"pure-text\" or \"multimodal\" !")
        self.sys_msg = open(prompt_path, "r", encoding="utf-8").read()

    def get_text_messages(self, question, texts, images):
        # 1. initialize system message
        messages = [{"role": "assistant", "content": self.sys_msg}]

        # 2. Add text quotes
        user_message = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            user_message += f"\n[{i + 1}] {text_str}"

        # 3. Add image quotes vlm-text or ocr-text
        user_message += "\nImage Quotes are:"
        for i, image in enumerate(images):
            img_description = image["img_description"]
            # img_description = image["img_ocr"]
            user_message += f"\nimage{i + 1} is described as: {img_description}"
        user_message += "\n\n"

        # 4. add user question
        user_message += f"The user question is: {question}"

        messages.append({"role": "user", "content": user_message})
        return messages

    def get_interleaved_messages(self, question, texts, images):
        # 1. initialize system message
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": self.sys_msg}]},  # system prompt
            {"role": "user", "content": []},  # user prompt is initialized as empty。
        ]
        # 2. Add text quotes
        text_quotes = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            text_quotes += f"\n[{i + 1}] {text_str}"
        messages[1]["content"].append({"type": "text", "text": f"{text_quotes}\n"})

        # 3. Add image quotes (actual image content)
        messages[1]["content"].append({"type": "text", "text": "Image Quotes are:\n"})
        for i, image in enumerate(images):
            messages[1]["content"].append({"type": "text", "text": f"- image{i + 1} is "})
            base64_image = encode_image(image["img_path"])
            messages[1]["content"].append(
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
            )

        # 4. Add user question
        messages[1]["content"].append({"type": "text", "text": f"The user question is: {question}"})
        return messages

    def get_api_response(self, q_id, question, texts, images):
        if self.mode == "pure-text":
            messages = self.get_text_messages(question, texts, images)
        else:
            messages = self.get_interleaved_messages(question, texts, images)

        try:
            completion = self.client.messages.create(model=self.model, max_tokens=1024, messages=messages)
            result = {"q_id": q_id,
                      "model": self.model,
                      "in_tok": completion.usage.input_tokens,
                      "out_tok": completion.usage.output_tokens,
                      "total_tok": completion.usage.input_tokens + completion.usage.output_tokens,
                      "response": completion.content[0].text,
                      }
        except Exception as e:
            result = {"q_id": q_id,
                      "model": self.model,
                      "in_tok": 0,
                      "out_tok": 0,
                      "total_tok": 0,
                      "response": "",
                      "error": str(e),
                      }
        return result


class Swift_Inference_PT:
    def __init__(self, model_id_or_path, lora_path=""):
        from swift.llm import PtEngine, get_template, RequestConfig
        # Get model and template, and load LoRA weights.
        if lora_path == "":
            engine = PtEngine(model_id_or_path)
            self.model = model_id_or_path
        else:
            engine = PtEngine(model_id_or_path, adapters=[lora_path])
            self.model = model_id_or_path + "_lora"
        self.sys_msg = open("prompt_bank/pure_text_infer.txt", "r", encoding="utf-8").read()
        template = get_template(engine.model_meta.template, engine.tokenizer, default_system=self.sys_msg)
        engine.default_template = template
        self.engine = engine
        self.request_config = RequestConfig(max_tokens=600, temperature=0.5)
        self.mode = "pure-text"

    @staticmethod
    def get_user_messages(question, texts, images):
        # 1. Add text quotes
        user_message = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            user_message += f"\n[{i + 1}] {text_str}"
        # 2. Add image quotes vlm-text or ocr-text
        user_message += "\nImage Quotes are:"
        for i, image in enumerate(images):
            img_description = image["img_description"]
            # img_description = image["img_ocr"]
            user_message += f"\nimage{i + 1} is described as: {img_description}"
        user_message += "\n\n"
        # 3. add user question
        user_message += f"The user question is: {question}"
        return [{'role': 'user', 'content': user_message}]

    def get_api_response(self, q_id, question, texts, images):
        from swift.llm import InferRequest
        user_message = self.get_user_messages(question, texts, images)
        infer_request = InferRequest(messages=user_message)
        completion = self.engine.infer([infer_request], self.request_config)
        result = {"q_id": q_id,
                  "model": self.model,
                  "in_tok": completion[0].usage.prompt_tokens,
                  "out_tok": completion[0].usage.completion_tokens,
                  "total_tok": completion[0].usage.total_tokens,
                  "response": completion[0].choices[0].message.content,
                  }
        return result


class Swift_Inference_VLLM:
    def __init__(self, model_id_or_path, lora_path=""):
        from swift.llm import VllmEngine, RequestConfig
        from swift.plugin import InferStats

        self.sys_msg = open("prompt_bank/multimodal_infer.txt", "r", encoding="utf-8").read()
        num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
        if lora_path == "":
            self.engine = VllmEngine(model_id_or_path,
                                     limit_mm_per_prompt={"image": 20},
                                     max_model_len=32768,
                                     tensor_parallel_size=num_gpu)
            self.model = model_id_or_path
        else:
            self.engine = VllmEngine(model_id_or_path,
                                     lora_path=lora_path,
                                     limit_mm_per_prompt={"image": 20},
                                     max_model_len=32768,
                                     tensor_parallel_size=num_gpu)
            self.model = model_id_or_path + "_lora"

        self.request_config = RequestConfig(max_tokens=512, temperature=0)
        self.metric = InferStats()
        self.mode = "multimodal"

    @staticmethod
    def get_user_messages(question, texts, images):
        img_list = []
        # 1. Add text quotes
        user_message = "Text Quotes are:"
        for i, text in enumerate(texts):
            text_str = text["text"]
            user_message += f"\n[{i + 1}] {text_str}"
        # 2. Add image quotes vlm-text or ocr-text
        user_message += "\nImage Quotes are:"
        for i, image in enumerate(images):
            user_message += f"- image{i + 1} is <image>"
            img_list.append(image["img_path"])
        user_message += "\n\n"
        # 3. add user question
        user_message += f"The user question is: {question}"
        return user_message, img_list

    def get_api_response(self, q_id, question, texts, images):
        from swift.llm import InferRequest
        user_msg, img_list = self.get_user_messages(question, texts, images)
        messages = {"messages": [{"role": "system", "content": self.sys_msg},
                                 {"role": "user", "content": user_msg}], "images": img_list}
        completion = self.engine.infer([InferRequest(**messages)],
                                       self.request_config,
                                       metrics=[self.metric])
        step = self.metric.compute()
        result = {"q_id": q_id,
                  "model": self.model,
                  "in_tok": step["num_prompt_tokens"],
                  "out_tok": step["num_generated_tokens"],
                  "total_tok": step["num_prompt_tokens"] + step["num_generated_tokens"],
                  "response": completion[0].choices[0].message.content,
                  }
        self.metric.reset()
        return result