import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]  # set cuda devices

from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import gc
import time

model_name = sys.argv[1] # enter model name, e.g., "HuggingFaceTB/SmolVLM-Instruct", etc.
isExt = int(sys.argv[2]) # 0 for original, 1 for extended
RANDOM_SEED = int(sys.argv[3]) # enter random seed for reproducibility
devices = sys.argv[4] # enter cuda device, e.g., "0,1,2,3", etc.

print("="*30)
print(f"model: {model_name}, isExt: {isExt}, seed: {RANDOM_SEED}, device(s): {devices}")

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
train_df = pd.read_json("../Data/Original/train.jsonl", lines=True)
if(isExt):
    df_train_ex = pd.read_json("../Data/Extended/ex_train.jsonl", lines=True)

    train_df = pd.concat([train_df, df_train_ex], ignore_index=True)

def balance_dataset(df, label_col='label'):
    # Split by class
    class_counts = df[label_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Sample from majority class to match minority count
    df_minority = df[df[label_col] == minority_class]
    df_majority = df[df[label_col] == majority_class].sample(n=len(df_minority), random_state=RANDOM_SEED)

    # Concatenate and shuffle
    balanced_df = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return balanced_df

train_df = balance_dataset(train_df)#.sample(n=200, random_state=RANDOM_SEED)

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df['label'])

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

train_dataset = train_df.apply(format_data, axis=1).tolist()
eval_dataset = val_df.apply(format_data, axis=1).tolist()
test_dataset = test_df.apply(format_data, axis=1).tolist()

# --- TRAINING ---
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name)

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=True,
    init_lora_weights="gaussian"
)

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    output_dir="./Chkpts/"+model_name.split("/")[-1]+str(isExt),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    warmup_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    report_to="none",
    max_length=None
)

class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Clear cache every 10 steps to prevent fragmentation
        if state.global_step % 10 == 0:
            print("Clearing CUDA cache to prevent fragmentation...")
            torch.cuda.empty_cache()
            gc.collect()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    # MEMORY FIX: Inject the cleaner callback
    callbacks=[ClearCacheCallback()]
)

trainer.train()
trainer.save_model(training_args.output_dir)

# --- EVALUATION ---
def clear_cuda_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

clear_cuda_memory()

# Load the fine-tuned model and processor
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_name)

adapter_path = "./Chkpts/"+model_name.split("/")[-1]+str(isExt)
model.load_adapter(adapter_path)

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
res.to_csv(f"Outputs/{model_name.split('/')[-1]}_isExt_{isExt}.csv", index=False)