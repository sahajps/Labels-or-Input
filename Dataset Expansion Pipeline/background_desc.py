import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os

device = 'cuda'

file = "train.jsonl"

df = pd.read_json("../Data/Original/"+file, lines=True)#.sample(500, random_state=42)
torch.set_grad_enabled(False)

# HF-model link: https://huggingface.co/internlm/internlm-xcomposer2d5-7b
model = AutoModel.from_pretrained('/home/sahajps/Models/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained('/home/sahajps/Models/internlm-xcomposer2d5-7b', trust_remote_code=True)

# AutoTokenizer uses the same token for padding (pad_token_id) and end-of-sequence (eos_token_id), which confuses the model when generating attention masks.
# ✅ Solution: Manually set the pad_token to a value different from eos_token.
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not set
#     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = '[PAD]'

model.tokenizer = tokenizer

query = "Analyze and describe the given image's visuals in short. Do not include the description about the text written on the image."

desc = []
for i in tqdm(df.img):
    image = ['Data/Original/'+i]
    with torch.autocast(device_type=device, dtype=torch.float16):
        response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, top_p=None, use_meta=True)
    desc.append(response)

df["desc"] = desc

os.makedirs('Outputs/', exist_ok=True)
df.to_json('Outputs/with_desc_'+file, orient='records', lines=True)