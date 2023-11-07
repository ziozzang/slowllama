import logging
import torch
import sys
import os

from loader import load_frozen
from utils import Tokenizer, greedy_gen
from conf_fp16 import *

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

lora_weights = sys.argv[1] if len(sys.argv) > 1 else None

# Use Original LLaMa or Modified foundation Model's Tokenizer
tokenizer = Tokenizer(llama2_model_path)

model = load_frozen(frozen_model_path, dropout=0.0, lora_rank=4, frozen_dtype=frozen_dtype, compute_dtype=compute_dtype).to(device)
if lora_weights is not None:
    logging.debug(model.load_state_dict(torch.load(lora_weights), strict=False))

logging.info('Model loaded.')

prompt = 'Cubestat reports the following metrics: '

greedy_gen(model, tokenizer, device, prompt, max_new_tokens=100)
