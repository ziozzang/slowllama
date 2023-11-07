import torch
from transformers import AutoTokenizer
import logging
from conf import *

def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)

def device_supports_dtype(device, dtype):
    try:
        tensor = torch.tensor([1.0, 2.0]).to(device).to(dtype)
        return True
    except TypeError as e:
        return False

global_id_auto = 0

def next_id():
    global global_id_auto
    new_id = global_id_auto
    global_id_auto += 1
    return new_id

def save_rng_state(device='cpu'):
    if device == 'cpu':
        import torch
        return torch.random.get_rng_state()
    elif device.startswith('cuda'):
        import torch.cuda
        return torch.cuda.get_rng_state(device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        return torch.mps.get_rng_state()
    else:
        raise ValueError(f"Unsupported device: {device}")

def restore_rng_state(rng_state, device='cpu'):
    if device == 'cpu':
        import torch
        torch.random.set_rng_state(rng_state)
    elif device.startswith('cuda'):
        import torch.cuda
        torch.cuda.set_rng_state(rng_state, device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        torch.mps.set_rng_state(rng_state)
    else:
        raise ValueError(f"Unsupported device: {device}")
    
def greedy_gen(model, tokenizer, device, prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        _, next_token = torch.topk(logits, k=1, dim=-1)
        logging.info(f'next token: {next_token} {tokenizer.decode(next_token.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for i, output in enumerate(tokens):
        logging.info(f'{i} - {tokenizer.decode(output.tolist())}')

def cleanup_cache(device='cpu'):
    if device.startswith('mps'):
        import torch.mps
        torch.mps.empty_cache()

    
class Tokenizer:
    def __init__(self, path):
        self.model = AutoTokenizer.from_pretrained(path)
        self.eos_token = self.model.eos_token
        self.bos_token = self.model.bos_token

    def encode(self, text, bos=False, eos=False):
        b = [self.model.bos_token_id] if self.model.bos_token else []
        e = [self.model.eos_token_id] if self.model.eos_token else []
        if debug_flag:
            print('>TOKENS:',self.model.tokenize(text))
        ret = self.model(text)['input_ids'] + e
        l_b = len(b) # Append BOS token front of...
        if b != ret[0:l_b]:
            ret = b + ret
        if debug_flag:
            print('>ENC:', ret)
        return ret

    def decode(self, tokens):
        if debug_flag:
            print('>DEC:', tokens[0])
        return self.model.decode(tokens[0])
