# #!/bin/bash

echo "openai/clip-vit-base-patch32 0 42"
python3 clip.py "openai/clip-vit-base-patch32" 0 42 "cuda"

echo "openai/clip-vit-base-patch32 1 42"
python3 clip.py "openai/clip-vit-base-patch32" 1 42 "cuda"

echo "Salesforce/blip-image-captioning-base 0 42"
python3 clip.py "Salesforce/blip-image-captioning-base" 0 42 "cuda"

echo "Salesforce/blip-image-captioning-base 1 42"
python3 clip.py "Salesforce/blip-image-captioning-base" 1 42 "cuda"