# #!/bin/bash

echo "HuggingFaceTB/SmolVLM-Instruct 42"
python3 vlm_zs.py "HuggingFaceTB/SmolVLM-Instruct" 0 "cuda"


eacho "Qwen/Qwen2.5-VL-3B-Instruct 42"
python3 vlm_zs.py "Qwen/Qwen2.5-VL-3B-Instruct" 0 "cuda"


eacho "Qwen/Qwen2-VL-7B-Instruct 42"
python3 vlm_zs.py "Qwen/Qwen2-VL-7B-Instruct" 0 "cuda"