import os

import torch

from utils import device_map, next_id, device_supports_dtype
from model_config import ModelArgs
from io import BytesIO
import atexit
from conf import *

# Optimze Memory Map for Lower Disk/SSD IO. (for GPU optimization/SSD Lifespan)
class SingletonInstane:
  __instance = None

  @classmethod
  def __getInstance(cls):
    return cls.__instance

  @classmethod
  def instance(cls, *args, **kargs):
    cls.__instance = cls(*args, **kargs)
    cls.instance = cls.__getInstance
    return cls.__instance

class ModelMMap(SingletonInstane):
    def __init__(self):
        self.d = {}
        self.readonly = False
    def get(self, k):
        if optimize_fileio:
          #if debug_flag:
          #  print('LM:G>', k)
          if k in self.d:
            #if debug_flag:
            #  print('LM:G>> RTN_INNER', k)
            return BytesIO(self.d[k])
          else:
            if os.path.exists(k):
                #if debug_flag:
                #    print('LM:G>> LOAD FROM FILE',k)
                with open(k, "rb") as fh:
                    self.d[k] = fh.read()
            else:
                #if debug_flag:
                #    print('LM:G>> Gen New',k)
                return BytesIO()
            return BytesIO(self.d[k])
        else:
          return open(k,'rb')
    def put(self, k, v):
        if optimize_fileio:
            #if debug_flag:
            #    print('LM:P>',k)
            self.d[k] = v.getbuffer()
            #print('LM:P/LEN>', len(self.d[k]))
        else:
          return open(k, 'wb')
    def set_readonly(self):
        self.readonly = True
    def dump(self):
        if self.readonly:
            return # nothing to do
        for i in self.d.keys():
            #if debug_flag:
            #    print('LM:E/D>', i)
            with open(i, "wb") as f:
                f.write(self.d[i])

# Dump mmap
def exit_handler():
    if optimize_fileio:
        mmaps = ModelMMap.instance()
        mmaps.dump()

atexit.register(exit_handler)

class BlackboxDisk(torch.nn.Module):
    def __init__(self, module, args: ModelArgs):
        super().__init__()
        self.mmaps = ModelMMap.instance()
        self.module_id = next_id()
        self.input_id = next_id()
        self.compute_dtype = args.compute_dtype
        self.served_model_path = args.served_model_path
        self.cached_data_path = args.cached_data_path
        # TODO: can we deduce this from the data itself
        self.frozen_dtype = args.frozen_dtype
        if args.init_frozen:
            t = BytesIO()
            torch.save(module.to('cpu').to(self.frozen_dtype), t)
            self.mmaps.put(self.frozen_path(), t)

    def frozen_path(self):
        folder = os.path.join(self.served_model_path, 'frozen')
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, f'block_{self.module_id}.pt')
    
    def input_path(self):
        folder = os.path.join(self.cached_data_path, 'inputs')
        if not os.path.exists(folder):
            os.makedirs(folder)
        return f'{folder}/saved_{self.input_id}.pt'

    def loaded_inner(self):
        t = self.mmaps.get(self.frozen_path())
        return torch.load(t, map_location='cpu')
    
    def load(self, device):
        if device_supports_dtype(device, self.frozen_dtype):
            t = self.mmaps.get(self.frozen_path())
            return torch.load(t, map_location=device_map(device)).to(self.compute_dtype)
        else:
            t = self.mmaps.get(self.frozen_path())
            res = torch.load(t, map_location='cpu')
            return res.to(self.compute_dtype).to(device_map(device))

    def save(self, module):
        t = BytesIO()
        torch.save(module.to('cpu').to(self.frozen_dtype), t)
        self.mmaps.put(self.frozen_path(), t)
    
    def load_input(self, device):
        t = self.mmaps.get(self.input_path())
        return torch.load(t, map_location=torch.device(device_map(device)))

    def forward(self, input, *args):
        t = BytesIO()
        torch.save(input, t)
        self.mmaps.put(self.input_path(), t)
        device = device_map(input.device)
        module = self.load(device)

        if not self.training:
            module.eval()
        
        # we offload model immediately anyway.
        # no need to have gradient here ever.
        with torch.no_grad():
            return module(input, *args)
