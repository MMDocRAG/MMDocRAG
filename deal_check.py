import re
import os
from data_utils import save_jsonl

from collections import defaultdict
dict1={"Qwen2.5-3B-Inst":"qwen2.5-3b",
       "Qwen2.5-3B-Inst-Fine-tuning":"qwen2.5-3b-ft",
       "Llama3.2-3B-Inst":"llama3.2-3b",
       "Qwen3-4B (think)":"qwen3-4b",
       "Mistral-7B-Inst":"mistral-7b",
       "Qwen2.5-7B-Inst":"qwen2.5-7b",
       "Qwen2.5-7B-Inst-Fine-tuning":"qwen2.5-7b-ft",
       "Llama3.1-8B-Inst":"llama3.1-8b",
       "Qwen3-8B (think)":"qwen3-8b",
       "Qwen2.5-14B-Inst":"qwen2.5-14b",
       "Qwen2.5-14B-Inst-Fine-tuning":"qwen2.5-14b-ft",
       "Qwen3-14B (think)":"qwen3-14b",
       "Mistral-Small-24B-Inst":"mistral-small-24b",
       "Qwen3-30B-A3B":"qwen3-30b-a3b",
       "Qwen2.5-32B-Inst":"qwen2.5-32b",
       "Qwen2.5-32B-Inst-Fine-tuning": "qwen2.5-32b-ft",
       "Qwen3-32B (think)":"qwen3-32b",
       "Mistral-8x7B-Inst":"mistral-8x7b",
       "Llama3.3-70B-Inst":"llama3.3-70b",
       "Qwen2.5-72B-Inst":"qwen2.5-72b",
       "Qwen2.5-72B-Inst-Fine-tuning":"qwen2.5-72b-ft",
       "Qwen3-235B-A22B":"qwen3-235b-a22b",
       "Deepseek-V3":"deepseek-v3",
       "Deepseek-R1":"deepseek-r1",
       "Deepseek-R1-Distill-Qwen-32B":"deepseek-r1-distill-qwen-32b",
       "Deepseek-R1-Distill-Llama-70B":"deepseek-r1-distill-llama-70b",
       "Qwen-Plus":"qwen-plus",
       "Qwen-Max":"qwen-max",
       "Gemini-1.5-Pro":"gemini-1.5-pro",
       "Gemini-2.0-Pro":"gemini-2.0-pro",
       "Gemini-2.0-Flash":"gemini-2.0-flash",
       "Gemini-2.0-Flash-Think":"gemini-2.0-flash-tk",
       "Gemini-2.5-Flash":"gemini-2.5-flash",
       "Gemini-2.5-Pro":"gemini-2.5-pro",
       "Claude-3.5-Sonnet":"claude-3.5-sonnet",
       "GPT-4-turbo":"gpt-4-turbo",
       "GPT-4o-mini":"gpt-4o-mini",
       "GPT-4o":"gpt-4o",
       "GPT-o3-mini":"gpt-o3-mini",
       "GPT-4.1-nano":"gpt-4.1-nano",
       "GPT-4.1-mini":"gpt-4.1-mini",
       "GPT-4.1":"gpt-4.1",
       "Janus-Pro-7B":"janus-pro-7b",
       "MiniCPM-o-2.6-8B":"minicpm-o-2.6-8b",
       "InternVL2.5-8B":"internvl2.5-8b",
       "InternVL3-8B":"internvl3-8b",
       "InternVL3-9B":"internvl3-9b",
       "InternVL3-14B":"internvl3-14b",
       "InternVL2.5-26B":"internvl2.5-26b",
       "InternVL2.5-38B":"internvl2.5-38b",
       "InternVL3-38B":"internvl3-38b",
       "InternVL2.5-78B":"internvl2.5-78b",
       "InternVL3-78B":"internvl3-78b",
       "Qwen2.5-VL-7B-Inst":"qwen2.5-vl-7b",
       "Qwen2.5-VL-32B-Inst":"qwen2.5-vl-32b",
       "Qwen2.5-VL-72B-Inst":"qwen2.5-vl-72b",
       "Qwen-VL-Plus":"qwen-vl-plus",
       "Qwen-VL-Max":"qwen-vl-max",
       "Qwen-QVQ-Max":"qwen-qvq-max",
       "Qwen-QwQ-Plus":"qwen-qwq-plus",
       "Llama4-Scout-17Bx16E":"llama4-scout-17b-16e",
       "Llama4-Mave-17Bx128E":"llama4-mave-17b-128e"}
model_type1={'pure-text': ['Qwen2.5-3B-Inst', 'Qwen2.5-3B-Inst-Fine-tuning', 'Llama3.2-3B-Inst', 'Qwen3-4B (think)', 'Qwen2.5-7B-Inst', 'Qwen2.5-7B-Inst-Fine-tuning', 'Mistral-7B-Inst', 'Llama3.1-8B-Inst', 'Qwen3-8B (think)', 'Qwen2.5-14B-Inst', 'Qwen2.5-14B-Inst-Fine-tuning', 'Qwen3-14B (think)', 'Mistral-Small-24B-Inst', 'Qwen3-30B-A3B', 'Qwen2.5-32B-Inst', 'Qwen2.5-32B-Inst-Fine-tuning', 'Qwen3-32B (think)', 'Mistral-8x7B-Inst', 'Llama3.3-70B-Inst', 'Qwen2.5-72B-Inst', 'Qwen2.5-72B-Inst-Fine-tuning', 'Qwen3-235B-A22B', 'Deepseek-V3', 'Deepseek-R1', 'Deepseek-R1-Distill-Qwen-32B', 'Deepseek-R1-Distill-Llama-70B', 'Qwen-Plus', 'Qwen-Max', 'Qwen-QwQ-Plus', 'Gemini-1.5-Pro', 'Gemini-2.0-Pro', 'Gemini-2.0-Flash', 'Gemini-2.0-Flash-Think', 'Gemini-2.5-Flash', 'Gemini-2.5-Pro', 'Claude-3.5-Sonnet', 'GPT-4-turbo', 'GPT-4o-mini', 'GPT-4o', 'GPT-o3-mini', 'GPT-4.1-nano', 'GPT-4.1-mini', 'GPT-4.1'], 'multi_modal': ['Janus-Pro-7B -', 'MiniCPM-o-2.6-8B -', 'InternVL2.5-8B', 'InternVL3-8B', 'InternVL3-9B', 'InternVL3-14B', 'InternVL2.5-26B', 'InternVL2.5-38B', 'InternVL3-38B', 'InternVL2.5-78B', 'InternVL3-78B', 'Qwen2.5-VL-7B-Inst', 'Qwen2.5-VL-32B-Inst', 'Qwen2.5-VL-72B-Inst', 'Llama4-Scout-17Bx16E', 'Llama4-Mave-17Bx128E', 'Qwen-VL-Plus', 'Qwen-VL-Max', 'Qwen-QVQ-Max', 'Gemini-1.5-Pro', 'Gemini-2.0-Pro', 'Gemini-2.0-Flash', 'Gemini-2.0-Flash-Think', 'Gemini-2.5-Flash', 'Gemini-2.5-Pro', 'Claude-3.5-Sonnet', 'GPT-4o-mini', 'GPT-4o', 'GPT-4.1-nano', 'GPT-4.1-mini', 'GPT-4.1']}


# 示例输入数据（替换为完整多行字符串或读取自文件）
raw_data1 = """
Qwen2.5-3B-Inst 3.6k 415 50.4 23.6 32.2 17.8 10.7 13.4 25.0 0.123 0.271 4.02 2.52 2.73 2.87 2.59 2.94
Qwen2.5-3B-Inst-Fine-tuning 3.6k 286 68.1 57.8 62.5 44.6 1.4 2.8 49.6 0.182 0.338 4.45 3.08 3.40 3.03 2.60 3.31
Llama3.2-3B-Inst 3.4k 418 37.9 25.7 30.6 18.5 30.4 23.0 23.0 0.089 0.243 3.51 2.62 2.79 2.38 2.57 2.84
Qwen3-4B (think) 3.6k 1072 67.6 65.4 66.5 22.8 65.5 33.8 48.6 0.065 0.175 4.25 3.13 3.57 3.55 3.40 3.58
Mistral-7B-Inst 4.0k 451 53.4 45.2 49.0 23.1 44.2 30.4 38.6 0.109 0.251 4.04 2.63 3.05 3.11 2.95 3.16
Qwen2.5-7B-Inst 3.6k 302 66.5 45.5 54.0 36.2 28.2 31.7 45.8 0.159 0.313 4.27 2.93 3.21 3.22 3.07 3.34
Qwen2.5-7B-Inst-Fine-tuning 3.6k 223 71.2 66.8 69.0 38.5 2.6 4.9 56.0 0.199 0.353 4.59 3.38 3.70 3.36 2.98 3.60
Llama3.1-8B-Inst 3.4k 435 54.1 51.8 52.9 24.1 38.1 29.5 41.0 0.112 0.254 4.17 2.88 3.15 3.08 2.99 3.25
Qwen3-8B (think) 3.6k 1018 70.7 68.5 69.6 22.5 82.9 35.4 48.2 0.064 0.174 4.17 3.16 3.40 3.38 3.22 3.47
Qwen2.5-14B-Inst 3.6k 362 71.5 56.0 62.8 34.8 43.9 38.8 54.7 0.148 0.295 4.26 3.15 3.48 3.33 3.24 3.49
Qwen2.5-14B-Inst-Fine-tuning 3.6k 282 74.1 70.6 72.3 53.0 6.4 11.5 59.4 0.212 0.366 4.69 3.62 3.93 3.64 3.34 3.84
Qwen3-14B (think) 3.6k 920 72.4 65.6 68.8 25.5 81.5 38.8 50.7 0.072 0.187 4.23 3.23 3.44 3.35 3.20 3.49
Mistral-Small-24B-Inst 3.7k 391 49.3 46.7 48.0 22.7 46.0 30.4 39.0 0.091 0.236 4.03 2.60 2.88 3.05 2.97 3.11
Qwen3-30B-A3B 3.6k 969 71.8 68.7 70.2 24.1 84.5 37.5 50.0 0.068 0.179 4.22 3.70 3.41 3.33 3.21 3.57
Qwen2.5-32B-Inst 3.6k 320 69.4 66.8 68.1 40.7 33.0 36.5 58.9 0.159 0.307 4.39 3.27 3.59 3.48 3.41 3.63
Qwen2.5-32B-Inst-Fine-tuning 3.6k 282 77.5 74.2 75.8 62.1 22.9 33.4 65.1 0.224 0.377 4.73 3.71 4.06 3.73 3.41 3.93
Qwen3-32B (think) 3.6k 917 72.1 58.6 64.7 26.2 83.3 39.8 48.3 0.070 0.187 4.24 3.20 3.50 3.33 3.25 3.50
Mistral-8x7B-Inst 4.0k 259 57.2 32.6 41.5 28.5 24.2 26.1 30.7 0.098 0.248 4.04 2.60 2.95 3.08 2.67 3.07
Llama3.3-70B-Inst 3.4k 430 54.3 82.5 65.5 30.6 64.3 41.5 55.6 0.120 0.264 3.93 2.72 3.17 3.11 3.26 3.24
Qwen2.5-72B-Inst 3.6k 380 76.5 62.1 68.5 38.8 49.2 43.4 59.1 0.173 0.324 4.48 3.41 3.71 3.64 3.53 3.75
Qwen2.5-72B-Inst-Fine-tuning 3.6k 286 76.6 74.8 75.7 56.9 23.4 33.1 64.9 0.225 0.377 4.76 3.74 4.11 3.78 3.48 3.97
Qwen3-235B-A22B 3.6k 1052 70.3 69.0 69.7 26.7 82.1 40.3 53.1 0.069 0.180 4.22 3.80 3.35 3.41 3.15 3.59
Deepseek-V3 3.4k 234 70.8 73.4 72.1 37.3 59.8 45.9 61.1 0.171 0.338 4.57 3.31 3.74 3.62 3.47 3.74
Deepseek-R1 3.4k 930 65.2 78.2 71.1 23.8 87.5 37.4 51.1 0.069 0.185 4.13 3.17 3.56 3.30 3.25 3.48
Deepseek-R1-Distill-Qwen-32B 3.6k 731 66.5 49.0 56.4 25.4 68.9 37.2 44.4 0.082 0.204 3.43 2.47 3.23 2.80 3.33 3.06
Deepseek-R1-Distill-Llama-70B 3.3k 680 70.1 53.8 60.9 27.5 70.8 39.6 48.3 0.088 0.209 3.77 2.65 3.40 2.82 3.35 3.20
Qwen-Plus 3.6k 316 70.2 62.5 66.1 36.2 53.1 43.1 55.4 0.169 0.318 4.35 3.28 3.57 3.51 3.44 3.63
Qwen-Max 3.6k 426 71.7 66.9 69.3 39.7 51.5 44.8 58.9 0.165 0.315 4.42 3.47 3.71 3.64 3.59 3.77
Qwen-QwQ-Plus
Gemini-1.5-Pro 3.6k 290 66.8 72.9 69.7 32.1 60.3 41.9 56.2 0.126 0.262 3.59 2.62 3.13 2.82 3.01 3.03
Gemini-2.0-Pro 3.6k 307 71.7 81.4 76.3 36.7 61.3 45.9 62.8 0.164 0.308 4.13 3.08 3.56 3.34 3.46 3.51
Gemini-2.0-Flash 3.6k 283 66.0 71.3 68.5 30.6 65.1 41.6 54.4 0.134 0.277 3.84 2.75 3.21 3.00 3.13 3.19
Gemini-2.0-Flash-Think 3.6k 275 72.0 73.6 72.8 37.4 60.5 46.2 61.0 0.133 0.272 4.14 3.04 3.54 3.27 3.35 3.47
Gemini-2.5-Flash 3.6k 385 67.4 81.7 73.8 29.9 79.9 43.5 59.5 0.131 0.268 4.02 3.09 3.68 3.39 3.57 3.55
Gemini-2.5-Pro 3.6k 387 71.3 87.5 78.6 35.7 78.5 49.1 65.1 0.142 0.281 4.25 3.35 3.94 3.64 3.77 3.79
Claude-3.5-Sonnet 3.8k 348 65.2 77.5 70.8 33.7 76.6 46.8 57.4 0.122 0.276 4.30 3.11 3.60 3.50 3.50 3.60
GPT-4-turbo 3.4k 353 69.9 63.6 66.6 36.8 51.4 42.9 57.7 0.148 0.304 4.28 3.15 3.44 3.46 3.47 3.56
GPT-4o-mini 3.4k 394 61.9 71.3 66.3 31.9 49.7 38.9 56.6 0.142 0.291 4.56 3.15 3.66 3.65 3.49 3.70
GPT-4o 3.4k 353 66.9 67.1 67.0 37.0 57.2 44.9 57.2 0.160 0.313 4.29 3.37 3.65 3.56 3.59 3.69
GPT-o3-mini 3.4k 623 71.1 66.0 68.5 31.2 52.2 39.0 55.3 0.086 0.204 3.46 2.73 3.23 2.93 3.13 3.10
GPT-4.1-nano 3.3k 320 62.1 40.0 48.7 27.2 46.6 34.4 40.8 0.129 0.285 3.15 2.77 3.31 3.00 3.04 3.05
GPT-4.1-mini 3.4k 411 66.8 80.6 73.0 30.6 68.8 42.3 61.0 0.137 0.283 4.46 3.45 3.98 3.81 3.78 3.90
GPT-4.1 3.4k 324 77.8 80.9 79.3 42.2 59.4 49.4 68.3 0.148 0.294 4.56 3.74 4.15 3.98 3.92 4.07
"""

raw_data2=""" 
Janus-Pro-7B  154 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.000 0.110 0.0 0.10 0.00 0.00 0.00 0.02
MiniCPM-o-2.6-8B  1346 13.0 11.5 12.2 13.9 19.4 16.2 9.3 0.062 0.184 2.13 1.74 1.33 2.27 1.29 1.75
InternVL2.5-8B 17.1k 182 38.1 38.9 38.5 16.8 2.3 4.1 33.0 0.085 0.269 4.05 2.67 2.80 2.87 2.77 3.00
InternVL3-8B 17.1k 419 61.8 30.3 40.7 27.3 46.5 34.4 37.0 0.119 0.260 4.01 2.63 3.01 3.04 2.71 3.08
InternVL3-9B 17.2k 287 72.4 52.4 60.8 33.9 25.9 29.3 50.9 0.146 0.303 4.31 2.65 3.14 3.17 3.08 3.27
InternVL3-14B 17.1k 369 66.5 51.7 58.1 27.4 56.9 37.0 49.9 0.149 0.292 4.29 2.63 3.12 3.18 3.03 3.25
InternVL2.5-26B 17.1k 198 56.8 26.6 36.3 21.9 5.4 8.6 25.8 0.094 0.291 3.90 2.51 2.77 2.80 2.75 2.95
InternVL2.5-38B 17.1k 470 25.2 40.1 31.0 24.5 11.5 15.7 31.3 0.098 0.257 4.04 2.70 2.88 2.75 2.73 3.02
InternVL3-38B 17.1k 359 67.7 51.0 58.2 33.1 64.7 43.8 53.9 0.155 0.301
InternVL2.5-78B 17.1k 357 63.1 23.0 33.7 34.1 32.6 33.4 26.6 0.127 0.301 3.93 2.54 2.80 2.79 2.75 2.96
InternVL3-78B
Qwen2.5-VL-7B-Inst 7.1k 128 58.0 14.5 23.2 31.3 11.0 16.3 16.6 0.069 0.273 4.05 1.75 1.89 2.36 2.29 2.47
Qwen2.5-VL-32B-Inst 7.0k 755 57.4 32.2 41.2 26.8 73.2 39.3 36.2 0.086 0.227 4.10 2.65 2.84 2.70 2.83 3.02
Qwen2.5-VL-72B-Inst 7.1k 320 68.9 72.1 70.5 36.0 52.9 42.8 57.5 0.151 0.298 4.15 3.08 3.43 3.35 3.33 3.47
Llama4-Scout-17Bx16E 11.6k 339 60.0 44.0 50.8 29.1 40.9 34.0 38.9 0.128 0.288 4.03 3.01 2.91 2.77 2.93 3.13
Llama4-Mave-17Bx128E 11.6k 320 69.6 74.2 71.8 41.8 30.8 35.5 58.6 0.151 0.308 4.25 3.29 3.63 3.55 3.61 3.67
Qwen-VL-Plus 7.1k 257 57.3 20.9 30.6 21.7 21.5 21.6 25.2 0.096 0.269 3.22 2.03 2.34 2.17 2.05 2.36
Qwen-VL-Max 7.1k 206 78.4 45.9 57.9 33.5 39.3 36.2 46.8 0.124 0.308 4.17 3.01 3.32 3.14 3.13 3.35
Qwen-QVQ-Max 6.8k 1137 59.3 11.8 19.7 26.0 45.0 33.0 24.6 0.064 0.179 3.20 2.01 2.37 2.11 2.03 2.34
Gemini-1.5-Pro 3.8k 202 68.0 72.5 70.2 36.8 45.6 40.7 59.3 0.098 0.261 3.27 2.50 2.90 2.48 2.68 2.77
Gemini-2.0-Pro 3.8k 265 69.1 82.4 75.1 36.0 61.3 45.3 62.0 0.148 0.298 3.91 2.87 3.33 3.12 3.30 3.31
Gemini-2.0-Flash 3.8k 226 72.8 69.7 71.2 37.8 63.4 47.4 60.0 0.130 0.292 3.69 2.79 3.17 2.86 3.05 3.11
Gemini-2.0-Flash-Think 3.8k 290 72.6 80.6 76.4 41.2 61.2 49.2 66.2 0.144 0.297 4.21 3.24 3.69 3.41 3.48 3.61
Gemini-2.5-Flash 3.7k 362 72.2 80.7 76.2 34.3 70.4 46.1 62.4 0.139 0.284 4.24 3.28 3.82 3.66 3.79 3.76
Gemini-2.5-Pro 3.7k 371 68.8 89.9 78.0 35.0 72.8 47.3 65.4 0.139 0.283 4.33 3.40 3.97 3.78 3.94 3.88
Claude-3.5-Sonnet 7.8k 313 68.9 82.7 75.2 35.6 68.9 46.9 62.5 0.120 0.279 4.25 3.22 3.71 3.54 3.53 3.65
GPT-4o-mini 8.5k 355 63.0 71.8 67.1 32.1 47.4 38.3 56.3 0.145 0.295 4.54 3.13 3.59 3.53 3.23 3.60
GPT-4o 6.4k 347 60.2 83.4 70.0 35.2 58.1 43.8 62.6 0.157 0.315 4.39 3.42 3.74 3.58 3.58 3.74
GPT-4.1-nano 14.2k 301 54.3 20.7 30.0 30.9 43.9 36.3 29.0 0.129 0.299 4.00 2.65 2.80 2.89 3.19 3.11
GPT-4.1-mini 9.8k 474 62.0 85.1 71.7 30.6 72.0 43.0 61.2 0.132 0.285 4.41 3.48 3.98 3.87 3.88 3.92
GPT-4.1 6.6k 306 77.2 84.5 80.7 42.9 66.0 52.0 70.2 0.157 0.313 4.61 3.75 4.20 4.10 4.04 4.14
"""




# 初始化两个字典

model_group = defaultdict(list)
answer_data_text= {}
answer_data_modal={}
# 遍历每一行，提取模型名和数值
for line in raw_data1.strip().split('\n'):
    parts = line.strip().split()
    model_tokens = []
    value_tokens = []

    for token in parts:
        if re.match(r'^-?\d+(\.\d+)?[kK]?$', token):
            value_tokens.append(token)
        else:
            model_tokens.append(token)

    model_name = ' '.join(model_tokens)
    model_group['pure-text'].append(model_name)

    # 将单位 k 转换为浮点数
    values = []
    for val in value_tokens:
        if val.lower().endswith('k'):
            values.append(float(val[:-1]) * 1000)
        else:
            values.append(float(val))

    # 保存最后 6 项指标
    answer_data_text[model_name] = values[-6:]

for line in raw_data2.strip().split('\n'):
    parts = line.strip().split()
    model_tokens = []
    value_tokens = []

    for token in parts:
        if re.match(r'^-?\d+(\.\d+)?[kK]?$', token):
            value_tokens.append(token)
        else:
            model_tokens.append(token)

    model_name = ' '.join(model_tokens)
    model_group['multi_modal'].append(model_name)

    # 将单位 k 转换为浮点数
    values = []
    for val in value_tokens:
        if val.lower().endswith('k'):
            values.append(float(val[:-1]) * 1000)
        else:
            values.append(float(val))

    # 保存最后 6 项指标
    answer_data_modal[model_name] = values[-6:]

# 打印检查结果


'''检测是否完全存在'''
for k, v in answer_data_text.items():
    model=dict1[k]
    response_file=f'response/{model}_pure-text_response_quotes20.jsonl'
    if not os.path.exists(response_file):
        print(f'response file {response_file} does not exist')
    llm_path=f'response/evaluation/{model}_pure-text_response_quotes20.jsonl_evaluation.jsonl'
    if not os.path.exists(llm_path):
        print(f'response file {llm_path} does not exist')

















for k, v in answer_data_modal.items():
    model=dict1[k]
    response_file=f'response/{model}_multimodal_response_quotes20.jsonl'
    if not os.path.exists(response_file):
        print(f'response file {response_file} does not exist')
    llm_path = f'response/evaluation/{model}_multimodal_response_quotes20.jsonl_evaluation.jsonl'
    if not os.path.exists(llm_path):
        print(f'response file {llm_path} does not exist')

