# UPC-HLE at SemEval-2025 Task 7: Multilingual Fact-Checked Claim Retrieval

## Overview
This repository contains the implementation of the system developed by **UPC-HLE** for **SemEval-2025 Task 7**, which focuses on **multilingual and cross-lingual fact-checked claim retrieval**. Our approach leverages **Text Embedding Models (TEMs)** combined with **Cross-Encoders (CEs)** for improved fact-check retrieval across languages.

## Features
- **Multilingual & Cross-Lingual Retrieval**: Supports claim retrieval across multiple languages.
- **Transformer-Based Models**: Uses **Multilingual-E5, JinaAI, and Arctic-Embed** for dense embeddings.
- **Re-Ranking with Cross-Encoders**: Fine-tuned **Jina Reranker v2** to improve ranking accuracy.
- **Lexical Filtering**: Inspired by CheckThat!2020, filters out irrelevant fact-checks before re-ranking.
- **Contrastive Learning**: Applies **negative sampling** to improve model robustness.
- **Data Handling**: Supports preprocessing, dataset splits, and evaluation scripts.

---
## Installation
### **1. Set up the environment**
This project uses **Conda**. You can create the environment using:
```sh
conda env create -f environment.yaml
conda activate semeval25
```
Alternatively, install dependencies using **pip**:
```sh
pip install -r requirements.txt
```

### **2. Download the Data**
Place the dataset in the `data/` directory. A sample dataset is included under `data/sample_data/` for quick testing.

### **3. Repository Structure**
```
├── README.md
├── environment.yaml       # Conda environment setup
├── requirements.txt       # Python dependencies
├── job_inference.sh       # Inference job script
├── job_train.sh           # Training job script
├── data/                  # Dataset and splits
│   ├── sample_data/       # Example dataset files
│   ├── splits/            # Predefined dataset splits
├── nbs/                   # Jupyter notebooks (data exploration, evaluation, error analysis)
├── scripts/               # Training and inference scripts
├── src/                   # Core codebase (models, preprocessing, evaluation)
└── models/                # Pretrained and fine-tuned models
```

---
## Training & Inference
### **Train Cross-Encoder (CE)**
Run the training script to fine-tune the cross-encoder model:
```sh
bash job_train.sh
```
Alternatively, use Python directly:
```sh
python scripts/contrastive/train.py --task_name crosslingual\
    --teacher_model_name 'path_to_model'\
    --reranker_model_name 'path_to_reranker'\
    --output_path 'path_to_output_folder'\
     --task_file data/splits/tasks.json
```

| Parameter Name        | Description |
|----------------------|-------------|
| **train_batch_size** | Number of training samples processed at once (NO NEED TO TUNE). |
| **num_epochs** | Number of times the model sees the entire dataset (TUNEABLE PARAMETER, MONITOR LOSS). |
| **dev_size_triplets** | Portion of data used for validation (NO NEED TO TUNE). |
| **pct_warmup** | Percentage of warmup steps before full training begins (PARTLY TUNEABLE). |
| **output_k** | Number of top retrieved candidates considered (NO NEED TO TUNE). |
| **optimizer_params** | Learning rate and other optimizer configurations (TUNEABLE PARAMETERS, EXPLORE MORE). |
| **emb_batch_size** | Batch size for embedding model processing (NO NEED TO TUNE). |
| **n_candidates** | Number of retrieved candidates before re-ranking (TUNEABLE PARAMETER). |
| **n_neg_candidates** | Number of negative candidates sampled for training (TUNEABLE, BEWARE OF UNBALANCE). |
| **neg_perc_threshold** | Score threshold to filter hard negatives (TUNEABLE PARAMETER). |


### **Run Inference**
To generate predictions on new claims:
```sh
bash job_inference.sh
```

---
## Evaluation
To evaluate the retrieval performance on test data:
```sh
python src/evaluate.py --dataset data/splits_test.json
```

**Metrics Used:**
- **Success@10** (Primary metric for ranking performance)
- **MRR (Mean Reciprocal Rank)**
- **Recall@10** (Additional retrieval effectiveness measure)

---
## Results
Our best-performing system achieved:
| Task               | Success@10 (Dev) | Success@10 (Test) |
|--------------------|------------------|------------------|
| **Monolingual**    | **0.91**          | **0.89**          |
| **Cross-Lingual**  | **0.78**          | **0.64**          |

We observe a drop in performance in the cross-lingual setting due to **missing context, translation inconsistencies, and dataset biases**.

**Comparison with Other Internal Systems:**
| System Name             | Dev Monolingual | Dev Cross-Lingual | Test Monolingual | Test Cross-Lingual |
|-------------------------|------------------|------------------|------------------|------------------|
| **mTEM-CE-Eng-SF**      | **0.91**         | **0.78**         | **0.89**         | **0.64**         |
| **mTEM-CE-E5**         | 0.90             | 0.75             | 0.91             | 0.75             |
| **mTEM-CE-Eng-E5**     | 0.90             | 0.77             | 0.90             | 0.76             |
| **mTEM-LF-CE-Eng-E5**  | 0.89             | 0.74             | 0.89             | 0.75             |
| **mTEM-jinav3**        | 0.83             | 0.57             | 0.83             | 0.54             |
| **mTEM-E5**            | 0.81             | 0.68             | 0.84             | 0.64             |
| **TEM-MP-QA**          | -                | -                | 0.74             | 0.52             |
| **LE-Eng-Trf**         | 0.63             | 0.34             | 0.66             | 0.40             |

---
## Ethical Considerations
While this system aims to **combat misinformation**, it has potential ethical risks:
1. **Bias in Fact-Check Coverage**: Some languages may have fewer fact-checks available, leading to disparities in retrieval quality.
2. **Misuse of Automation**: Fully automated fact-checking systems could be misused for **censorship** or **biased content filtering**.
3. **False Positives/Negatives**: Incorrect retrievals could mislead users if not properly validated by human experts.

To mitigate these risks, **human verification is recommended** when deploying this system.

---
## Citation
If you use this work, please cite our paper:
```bibtex
@inproceedings{semeval2025task7upchle,
    title={UPC-HLE at SemEval-2025 Task 7: Multilingual Fact-Checked Claim Retrieval with Text Embedding Models and Cross-Encoder Re-Ranking},
    author={Alberto Becerra-Tome, Agustín Conesa},
    booktitle = {Proceedings of the 19th International Workshop on Semantic Evaluation},
    series = {SemEval 2025},
    year = {2025},
    address = {Vienna, Austria},
    month = {July},
    pages = {}, %leave blank
    doi= {} %leave blank
}

```

---
## Code & Model Access
- **Code**: [GitHub Repository](https://github.com/BecTome/semeval25-FactCheckedClaimRetrieval)
- **Models**: [Hugging Face](https://huggingface.co/UPC-HLE)

---
## Contact
For any questions or contributions, feel free to reach out:
📧 **Alberto Becerra-Tome** - alberto.becerra1@bsc.es  
📧 **Agustin Conesa** - agustin.conesa.celdran@estudiantat.upc.edu
