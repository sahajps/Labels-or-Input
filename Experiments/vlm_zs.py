import os
import sys
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np

model_name = sys.argv[1] # enter model name, e.g., "HuggingFaceTB/SmolVLM-Instruct", etc.
RANDOM_SEED = int(sys.argv[2]) # enter random seed for reproducibility
device = sys.argv[3] # enter cuda device, e.g., "cuda:0", etc.

print("="*30)
print(f"model: {model_name}, seed: {RANDOM_SEED}, device(s): {device}")

def seed_setter(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    set_seed(seed)

seed_setter(RANDOM_SEED)

# Load your data
test_df_seen = pd.read_json("../Data/Original/test_seen.jsonl", lines=True)
test_df_unseen = pd.read_json("../Data/Original/test_unseen.jsonl", lines=True)
test_df = pd.concat([test_df_seen, test_df_unseen], ignore_index=True)

# Resize very large images
def resize_image(image, max_size):
    width, height = image.size
    
    # Only resize if the image is actually larger than the max_size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

# Format data for VLM
def format_data(sample):
    try:
        img_pil = Image.open("../Data/Original/"+sample["img"]).convert("RGB")
    except:
        img_pil = Image.open("../Data/Extended/"+sample["img"]).convert("RGB")

    img_pil = resize_image(img_pil, max_size=768)

    return {
      "images": [img_pil],
      "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pil,
                },
                {
                    "type": "text",
                    "text": f"""You are given a meme image and its text caption. Determine if the meme is hateful. Respond with "Yes" or "No" only.
Caption: '{sample["text"]}'
isHateful:""",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": ("Yes." if sample["label"]==1 else "No."),
                }
            ],
        },
        ]
    }

test_dataset = test_df.apply(format_data, axis=1).tolist()

# --- MODEL LOADING ---
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=device,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name)

# --- EVALUATION ---
def generate_text_batch(samples, max_new_tokens=3):
    text_inputs = []
    image_inputs = []
    
    for sample in samples:
        # Apply chat template to each sample
        text_input = processor.apply_chat_template(
            sample['messages'][0:-1],  # Use the sample without the output
            add_generation_prompt=True
        )
        text_inputs.append(text_input)
        
        # Collect image for each sample
        image = sample['images']
        image_inputs.append(image)
    
    # Prepare the inputs for the model (batch processing)
    model_inputs = processor(
        text=text_inputs,
        images=image_inputs,
        return_tensors="pt",
        padding=True,  # Important for batch processing
        padding_side="left"
    ).to(model.device)
    
    # Generate text with the model
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False
    )
    
    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the output text for all samples
    output_texts = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Return list of cleaned outputs
    return [text.strip().lower() for text in output_texts]

# Generating predictions for the test dataset
preds_labels, true_labels = [], []
output_text = []
bs = 16
for i in tqdm(range(0, len(test_dataset), bs)):
    batch_samples = test_dataset[i:i+bs]
    batch_outputs = generate_text_batch(batch_samples, max_new_tokens=3)
    output_text.extend(batch_outputs)

for i in range(len(test_dataset)):
    # output_text = generate_text_from_sample(sample, max_new_tokens=3)
    if "yes" in output_text[i]:
        preds_labels.append(1)
    elif "no" in output_text[i]:
        preds_labels.append(0)
    else:
        preds_labels.append(2)  # if unclear

    true_text = test_dataset[i]['messages'][-1]['content'][0]['text'].lower()
    if "yes" in true_text:
        true_labels.append(1)
    elif "no" in true_text:
        true_labels.append(0)
    else:
        true_labels.append(2)  # if unclear

# --- Printing Results ---
print("\n\n***** Printing Results *****\n\n")
print(f"Classification Report:\n{classification_report(true_labels, preds_labels)}", end="\n\n")

cm = confusion_matrix(true_labels, preds_labels)
print(f"Confusion Matrix:\n{cm}", end="\n\n")

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

print(f"False Positive Rate (FPR): {fpr:.4f}", end="\n")
print(f"False Negative Rate (FNR): {fnr:.4f}", end="\n\n")

print("="*25, end="\n\n\n\n")

# --- Saving Test Predictions ---
os.makedirs("Outputs", exist_ok=True)
res = pd.DataFrame()
res['true_label'] = true_labels
res['predicted_label'] = preds_labels
res.to_csv(f"Outputs/{model_name.split('/')[-1]}_ZS.csv", index=False)