import pandas as pd
from utils import Gemini_load_balancer
from tqdm import tqdm
import os

df = pd.read_json("Outputs/with_new_text_caption.jsonl", lines=True)

def image_path_changer(path):
    if isinstance(path, str):
        path = path.replace("img/", "img/ex_")
        if ".jpg" in path:
            path = path.replace(".jpg", ".png")
        elif ".jpeg" in path:
            path = path.replace(".jpeg", ".png")
    return path

df["img"] = df["img"].apply(image_path_changer)

def extra_inverted_comma_remover(text):
    if text[0] == '"' or text[0] == "'":
        text = text[1:]
    if text[-1] == '"' or text[-1] == "'":
        text = text[:-1]

    return text

df.new_text = df.new_text.apply(extra_inverted_comma_remover)

df.text = df.new_text
df.drop(["new_text"], axis=1, inplace=True)
df.label = df.label.map(
    {0: 0, 1: 0}
)  # as we are generating non-hateful memes, we set all labels to 0

for id, txt, desc in tqdm(zip(df.id, df.text, df.desc)):
    prompt = f"You are a creative social media expert with deep knowledge of internet memes. Your task is to generate a humorous and engaging meme image that is non-hateful. Use the given background description and overlay the provided text in a visually appealing way. \n\nBackground image description: '{desc}' \n\nMeme text: '{txt}' \n\nEnsure the meme aligns with popular internet humor while remaining appropriate and widely relatable."

    tmp = id.split(".")[0]
    Gemini_load_balancer(prompt, "gemini-2.0-flash-exp", id=tmp, save_path="../Data/Extended/img/")

check_files = []
for id in df.id:
    tmp = id.split(".")[0]
    check_files.append(os.path.exists("../Data/Extended/img/ex_" + tmp + ".png"))

df = df[check_files]
df.drop(["desc"], axis=1, inplace=True)

df.to_json("../Data/Extended/ex_train.jsonl", orient="records", lines=True)
