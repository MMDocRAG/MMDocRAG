import base64
import json


def encode_image(image_path): #  base 64 编码格式
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_jsonl(data_list, filename):
    with open(filename, 'w', encoding="utf-8") as file:
        for data in data_list:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')


def load_jsonl(filename, debug_mode=False):
    data_list = []
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            if not debug_mode:
                data_list.append(json.loads(line.strip()))
            else:
                try:
                    data_list.append(json.loads(line.strip()))
                except:
                    print(line.strip())
    return data_list


def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
