# CS308 Labs 11–12 · Eduge text classification

Open [Eduge SVM and plagiarism detection](Eduge_SVM_baseline.ipynb) or [transformer text classification](EAI_lab_Eduge.ipynb) to see the saved outputs and plots. The files ending in `_source.ipynb` are the supplied notebooks.

## Rerun the labs

Put your original course file `textclassification.zip` at the repository root and start Jupyter from the root or the `lab11_12` folder. The notebooks extract the required files into `.work/lab11_12/`. The transformer notebook also caches its pretrained models under `.work/huggingface/`.

The notebooks were run with Python 3.12, PyTorch 2.11, Transformers 4.57.3, Datasets 4.8.5, Accelerate 1.16.0, and SentencePiece 0.2.3. They also use scikit-learn, NumPy, pandas, matplotlib, seaborn, and Jupyter. The transformer experiments used CUDA with bfloat16 on an RTX 5070 Ti. The SVM models are fitted when the notebook runs.

## Use the trained classifier on another machine

Transfer `eduge_bert_inference.zip` separately and extract it. It contains the fine-tuned Mongolian BERT weights, tokenizer, config, and nine-label mapping. The classifier can run on CPU and does not require the training dataset or retraining.

```python
from transformers import AutoTokenizer, pipeline

model_dir = "./eduge_bert_final"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
classify = pipeline(
    "text-classification",
    model=model_dir,
    tokenizer=tokenizer,
    device=-1,
)
print(classify("Монголын шигшээ баг хожлоо.", truncation=True, max_length=256))
```

Running the transformer notebook saves the checkpoint under `.work/lab11_12/eduge_bert_final/`. The 256-token example above uses the baseline input strategy. The optional Head+Tail strategy is implemented in the notebook.

Original course archive SHA-256: `1f126f3ac0e705ca51476ceaaff7fd8df036efd7e34f1af2851966b0f6972e91`.
