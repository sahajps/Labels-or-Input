import pandas as pd
from tqdm import tqdm
from utils import OpenAI_model_response

df = pd.read_json("Outputs/data_for_regeneration.jsonl", lines=True)#.sample(10, random_state=13)

new_text = [] # which will be non-hateful wrt to current desc of background
for cap, desc in tqdm(zip(df.text, df.desc)):
    prompt = f"""You are a helpful social media expert with deep knowledge of memes and inclusive language. Your task is to rewrite a hateful meme caption so that it becomes non-hateful and bias-reducing, while staying relevant to the background image description.

The revised caption must:
- Preserve reference to the originally targeted group or person (e.g., Muslims, Women, Immigrant, etc.), if they are mentioned. The new caption should try to retain the word related to target group in a respectful and bias-correcting way.
- Transform the message to be neutral or positively framed toward the mentioned group/person.
- Help counteract harmful stereotypes or biases.
- Remain coherent and contextually appropriate for the image.

Strict Rule: Only return the revised caption. Do not include explanations, disclaimers, or any other text.

Meme background image description: '{desc}'  

Meme caption text: '{cap}'  

Revised non-hateful meme caption:"""
    new_text.append(OpenAI_model_response(prompt, "gpt-4o-mini"))

df["new_text"] = new_text
df.to_json("Outputs/with_new_text_caption.jsonl", orient='records', lines=True)