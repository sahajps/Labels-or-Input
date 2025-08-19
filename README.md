# Labels or Input 🤔
📄 ***Abstract:*** The modern web is saturated with multimodal content, intensifying the challenge of detecting hateful memes, where harmful intent is often conveyed through subtle interactions between text and image under the guise of humor or satire. While recent advances in Vision-Language Models (VLMs) show promise, these models lack support for fine-grained supervision and remain susceptible to implicit hate speech. In this paper, we present a dual-pronged approach to improve multimodal hate detection. First, we propose a prompt optimization framework that systematically varies prompt structure, supervision granularity, and training modality. We show that prompt design and label scaling both influence performance, with structured prompts improving robustness even in small models, and InternVL2 achieving the best F1-scores across binary and scaled settings. Second, we introduce a multimodal data augmentation pipeline that generates 2,479 counterfactually neutral memes by isolating and rewriting the hateful modality. This pipeline, powered by a multi-agent LLM–VLM setup, successfully reduces spurious correlations and improves classifier generalization. Our approaches inspire new directions for building synthetic data to train robust and fair vision-language models. Our findings demonstrate that prompt structure and data composition are as critical as model size, and that targeted augmentation can support more trustworthy and context-sensitive hate detection.

> **Note:** This repository contains the code and data for the **multi-modal (MM) data augmentation** component of our paper. For the MM **label scaling** code, refer to this repo: [Multi-Modal-Scale](https://github.com/reycn/Multi-Modal-Scale).

## 📦 Dataset
We use two datasets (publically available on 🤗):  🧩 [Original Dataset](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)  ***&*** 🧪 [Extended Dataset](https://huggingface.co/datasets/sahajps/Meme-Sanity) (Ours). Instructions for dataset placement in `Data` folder are available [here](https://github.com/sahajps/Labels-or-Input/blob/main/Data/README.md).

## 🚀 Run
Ensure you're using `python 3.11.11` and install dependencies:
```bash
pip install -r requirements.txt
```
### 🧬 Dataset Expansion Pipeline
To run the MM data augmentation pipeline:
```bash
cd Dataset\ Expansion\ Pipeline
bash run.sh
```
> **Note:** Due to reliance on OpenAI and Gemini APIs, results may not be fully reproducible.  
> However, intermediate outputs are available in the `Dataset Expansion Pipeline/Output` folder for better understanding and debugging.

Generated data is saved in `Data/Extended` (also hosted on 🤗).
### 📊 Benchmarking
Run experiments for various modalities:
```bash
cd Experiments
bash run_text.sh
bash run_image.sh
bash run_clip.sh
```
Predicted test labels are stored in the `Outputs` folder.  To compute evaluation metrics, use: `Experiments/analysis.ipynb`.

### 👥 Human Evaluation
To assess the quality of augmented data, run the notebook `Human Eval/human_scoring.ipynb`.

## 📚 Cite Us
If you find this work helpful and use our dataset or methodology, please cite:
```bibtex
@misc{singh2025labelsinputrethinkingaugmentation,
  title={Labels or Input? Rethinking Augmentation in Multimodal Hate Detection},
  author={Sahajpreet Singh and Rongxin Ouyang and Subhayan Mukerjee and Kokil Jaidka},
  year={2025},
  eprint={2508.11808},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.11808},
}
```
For the initial MM label scaling root work:
```bibtex
@inproceedings{ouyang2025hateful,
  title={Hateful Meme Detection through Context-Sensitive Prompting and Fine-Grained Labeling (Student Abstract)},
  author={Ouyang, Rongxin and Jaidka, Kokil and Mukerjee, Subhayan and Cui, Guangyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={28},
  pages={29459--29461},
  year={2025}
}
```
## ❓ Issues
Open an issue on GitHub or reach out via email. We welcome questions and contributions! Thank you.