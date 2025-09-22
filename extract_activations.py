import json
import pickle
import random
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from einops import rearrange
from IPython.display import display, HTML
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor

from baukit import Trace, TraceDict

warnings.filterwarnings("ignore")

def extract_attention_head_activations(model, statements):
    if model.config.architectures[0] == "SmolVLMForConditionalGeneration":
        HEADS = [f"model.text_model.layers.{i}.self_attn.head_out" for i in range(model.model.text_model.config.num_hidden_layers)]
    else:
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    print(f"Extracting activations for {len(HEADS)} hidden layers")
    head_wise_hidden_states_list = []
    for prompt in tqdm(statements, total=len(statements)):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                # print(ret)
                _ = model(prompt.to('cuda'), output_hidden_states=True, output_attentions=True)
                head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
                head_wise_hidden_states_list.append(head_wise_hidden_states[:, :, :])
    features = []
    for head_wise_hidden_states in head_wise_hidden_states_list:
        if model.config.architectures[0] == "SmolVLMForConditionalGeneration":
            features.append(rearrange([np.array(head_wise_hidden_states[:,-1,:])], 'b l (h d) -> b l h d', h = model.model.text_model.config.num_attention_heads))
        else:
            features.append(rearrange([np.array(head_wise_hidden_states[:,-1,:])], 'b l (h d) -> b l h d', h = model.config.num_attention_heads))
    features = np.stack(features, axis=0)
    return features

# get model class
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# from custom_models.qwen2 import Qwen2ForCausalLM
# model_class = Qwen2ForCausalLM

model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
from custom_models.smolvlm import SmolVLMForConditionalGeneration
model_class = SmolVLMForConditionalGeneration

# tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", use_fast=True)
# statements = [
#     tokenizer("USER: This is a test statement.\nASSISTANT:", return_tensors="pt")['input_ids'],
# ]*10
processor = AutoProcessor.from_pretrained(model_name)
messages = [[
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]]*10

inputs = [processor.apply_chat_template(
	message,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to("cuda")["input_ids"] for message in messages]
model = model_class.from_pretrained(model_name, low_cpu_mem_usage=True, dtype=torch.float16, device_map="auto", attn_implementation="eager")
# print(model.config)
# print(model.model.text_model)
features = extract_attention_head_activations(model, inputs)
# (10, 1, [32: num_layers], 32, 128) for Llama 8B
# (10, 1, [80: num_layers], 64, 128) for Llama 70B
# (10, 1, [28: num_layers], 28, 128) for Qwen 7B
# (10, 1, [28: num_layers], 12, 128) for Qwen 1.5B
# (10, 1, [48: num_layers], 40, 128) for Qwen 14B
print(features.shape)  