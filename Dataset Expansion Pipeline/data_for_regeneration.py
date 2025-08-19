import pandas as pd

df = pd.read_json("Outputs/with_hatelabels_with_desc_train.jsonl", lines=True)

def label_map(x):
    x = x.lower()
    if 'yes' in x:
        return 1
    elif 'no' in x:
        return 0
    # output in some other format
    return None

df.hateLabelText = df.hateLabelText.apply(label_map)
df.hateLabelDesc = df.hateLabelDesc.apply(label_map)

# taking the case of harmful caption and harmless description
df = df[(df.label==1) & (df.hateLabelText==1) & (df.hateLabelDesc==0)]

df = df[["id", "img", "label", "text", "desc"]]
df.to_json("Outputs/data_for_regeneration.jsonl", orient='records', lines=True)