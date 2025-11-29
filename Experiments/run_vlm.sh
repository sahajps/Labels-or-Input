# #!/bin/bash

echo "HuggingFaceTB/SmolVLM-Instruct 0 42"
python3 vlm.py "HuggingFaceTB/SmolVLM-Instruct" 0 42 7

echo "HuggingFaceTB/SmolVLM-Instruct 1 42"
python3 vlm.py "HuggingFaceTB/SmolVLM-Instruct" 1 42 4

eacho "Qwen/Qwen2.5-VL-3B-Instruct 0 42"
python3 vlm.py "Qwen/Qwen2.5-VL-3B-Instruct" 0 42 5

echo "Qwen/Qwen2.5-VL-3B-Instruct 1 42"
python3 vlm.py "Qwen/Qwen2.5-VL-3B-Instruct" 1 42 6

eacho "Qwen/Qwen2-VL-7B-Instruct 0 42"
python3 vlm.py "Qwen/Qwen2-VL-7B-Instruct" 0 42 5

echo "Qwen/Qwen2-VL-7B-Instruct 1 42"
python3 vlm.py "Qwen/Qwen2-VL-7B-Instruct" 1 42 6