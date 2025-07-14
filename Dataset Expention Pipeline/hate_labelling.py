import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = 'cuda'
model_name = 'Qwen-2.5'
file = 'with_desc_train.jsonl'

model_list = {'Qwen-2.5': '/home/sahajps/Models/Qwen2.5-14B-Instruct-1M'}

model = AutoModelForCausalLM.from_pretrained(
    model_list[model_name],
    torch_dtype=torch.float16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_list[model_name], padding_side='left')
tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token

df = pd.read_json("Outputs/"+file, lines=True)

hateLabelText = []
hateLabelDesc = []
for txt, des in tqdm(zip(df.text, df.desc), total=len(df)):
    prompt_t = "You are a social media content moderator. You are given a text caption of a meme and your task is to identify whether it's hateful? Answer in 'Yes' or 'No' only. \n\n>>> Image Caption: "+ txt + " \n\n>>> isHateful:"

    prompt_d = "You are a social media content moderator. You are given a text description of the background image of a meme and your task is to identify whether it's hateful? Answer in 'Yes' or 'No' only. \n\n>>> Image Description: " + des + " \n\n>>> isHateful:"

    prompt = [prompt_t, prompt_d]

    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # by default, do_sample is True for Qwen-2.5 model
        outputs = model.generate(**inputs, max_new_tokens=1)

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for j in range(len(outputs)):
        outputs[j] = outputs[j][len(prompt[j]):].strip()

    hateLabelText.append(outputs[0])
    hateLabelDesc.append(outputs[1])

    del inputs, outputs
    torch.cuda.empty_cache()

df["hateLabelText"] = hateLabelText
df["hateLabelDesc"] = hateLabelDesc

df.to_json("Outputs/with_hatelabels_"+file, orient='records', lines=True)