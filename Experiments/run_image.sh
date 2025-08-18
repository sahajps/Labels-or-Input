# #!/bin/bash

echo "google/vit-base-patch16-224 0 42"
python3 image_only.py "google/vit-base-patch16-224" 0 42 "cuda"

echo "google/vit-base-patch16-224 1 42"
python3 image_only.py "google/vit-base-patch16-224" 1 42 "cuda"

echo "facebook/deit-base-patch16-224 0 42"
python3 image_only.py "facebook/deit-base-patch16-224" 0 42 "cuda"

echo "facebook/deit-base-patch16-224 1 42"
python3 image_only.py "facebook/deit-base-patch16-224" 1 42 "cuda"

echo "microsoft/beit-base-patch16-224-pt22k-ft22k 0 42"
python3 image_only.py "microsoft/beit-base-patch16-224-pt22k-ft22k" 0 42 "cuda"

echo "microsoft/beit-base-patch16-224-pt22k-ft22k 1 42"
python3 image_only.py "microsoft/beit-base-patch16-224-pt22k-ft22k" 1 42 "cuda"