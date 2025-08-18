# #!/bin/bash

echo "google-bert/bert-base-uncased 0 42"
python3 text_only.py "google-bert/bert-base-uncased" 0 42 "cuda"

echo "google-bert/bert-base-uncased 1 42"
python3 text_only.py "google-bert/bert-base-uncased" 1 42 "cuda"

echo "FacebookAI/roberta-base 0 42"
python3 text_only.py "FacebookAI/roberta-base" 0 42 "cuda"

echo "FacebookAI/roberta-base 1 42"
python3 text_only.py "FacebookAI/roberta-base" 1 42 "cuda"

echo "albert/albert-base-v2 0 42"
python3 text_only.py "albert/albert-base-v2" 0 42 "cuda"

echo "albert/albert-base-v2 1 42"
python3 text_only.py "albert/albert-base-v2" 1 42 "cuda"