# MMDocRAG Dataset



## :low_brightness:Overview

**MMDocRAG** 




## :high_brightness: ​Annotation Format

Our annotations are provided in json-line format, each corresponding to annotation as follows: QA pair, 15/20 quotes and multimodal generation:

- [dev_15.jsonl](https://github.com/MMDocRAG/MMDocRAG/blob/main/dataset/dev_15.jsonl): 2,055 json-lines, with 15 quotes for each line.

- [dev_20.jsonl](https://github.com/MMDocRAG/MMDocRAG/blob/main/dataset/dev_20.jsonl): 2,055 json-lines, with 20 quotes for each line.

- [evaluation_15.jsonl](https://github.com/MMDocRAG/MMDocRAG/blob/main/dataset/evaluation_15.jsonl): 2,000 json-lines, with 15 quotes for each line.

- [evaluation_20.jsonl](https://github.com/MMDocRAG/MMDocRAG/blob/main/dataset/evaluation_20.jsonl): 2,000 json-lines, with 20 quotes for each line.

The detailed formats are:

| Name                     | Type   | Description                                                  |
| ------------------------ | ------ | ------------------------------------------------------------ |
| `q_id`                   | int    | question id                                                  |
| `doc_name`               | string | document name                                                |
| `domain`                 | string | document's domain or category                                |
| `question`               | string | question annotation                                          |
| `evidence_modality_type` | list[] | evidence modalities                                          |
| `question_type`          | string | question type                                                |
| `text_quotes`            | list[] | candidate list of text quotes                                |
| `img_quotes`             | list[] | candidate list of image quotes                               |
| `gold_quotes`            | list[] | list of gold quotes, e.g., `["image1", "text2", "image3", "text9"]` |
| `answer_short`           | string | short-formed answer                                          |
| `answer_interleaved`     | string | multimodal answer, interleaving text with images             |

Each `text_quotes` or  `img_quotes` item consists of :

| Name              | Type   | Description                                                  |
| ----------------- | ------ | ------------------------------------------------------------ |
| `quote_id`        | string | quote identifier, e.g., `"image1", ..., "image8", "text1", ..., "text12" |
| `type`            | string | quote type, either "image" or "text"                         |
| `text`            | string | raw text for text quotes                                     |
| `img_path`        | string | image path for image quotes                                  |
| `img_description` | string | image description for image quotes                           |
| `page_id`         | int    | page identifier of this quote                                |
| `layout_id`       | int    | layout identifier of this quote                              |



## :high_brightness: ​Training Format

The training set ([train.jsonl](https://github.com/MMDocRAG/MMDocRAG/blob/main/dataset/train.jsonl)) consist of 4,110 json-lines, each following the OpenAI message format in <`system`, `user`, `assistant`> triplets:

| Role      | Content                                                      |
| --------- | ------------------------------------------------------------ |
| system    | prompt message for multimodal generation task, specified in [pure_text_infer.txt](https://github.com/MMDocRAG/MMDocRAG/blob/main/prompt_bank/pure_text_infer.txt) |
| user      | consolidate user inputs as follows: `text_quotes`, `img_quotes`, and `question` |
| assistant | ground truth annotations, i.e., `answer_interleaved`         |
