### Data Setup

Place the following files in the `Data/Original/` directory:

**Dataset source:** [https://huggingface.co/datasets/limjiayi/hateful_memes_expanded](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)

**Folder structure:**
```
Data/Original/
├── img/*                 # Meme image files
├── train.jsonl           # Training data
├── dev_seen.jsonl        # NOT USED: Validation set (seen)
├── dev_unseen.jsonl      # NOT USED: Validation set (unseen)
├── test_seen.jsonl       # Test set (seen)
├── test_unseen.jsonl     # Test set (unseen)
```

Place the following files in the `Data/Extended/` directory:

**Dataset source:** [https://huggingface.co/datasets/sahajps/Meme-Sanity](https://huggingface.co/datasets/sahajps/Meme-Sanity)

**Folder structure:**
```
Data/Original/
├── img/*                 # Meme image files
├── ex_train.jsonl        # Extended Augmented Training data
```