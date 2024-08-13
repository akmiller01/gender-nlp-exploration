# ! pip install datasets evaluate transformers accelerate huggingface_hub --quiet

# from huggingface_hub import login

# login()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import evaluate
import numpy as np


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/curated-climate-weighted")
# MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-climate-multi-classifier-weighted2")
MODEL = MODEL.to(DEVICE)


unique_labels = [
    "Significant climate adaptation objective",
    "Principal climate adaptation objective",
    "Significant climate mitigation objective",
    "Principal climate mitigation objective"
]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}


def preprocess_function(example):
    adaptation_label = example['adaptation_label']
    mitigation_label = example['mitigation_label']
    labels = [0. for i in range(len(unique_labels))]
    for unique_label in unique_labels:
        if adaptation_label == unique_label:
            labels[label2id[unique_label]] = 1.
            # if adaptation_label == "Principal climate adaptation objective":
            #     labels[label2id["Significant climate adaptation objective"]] = 1.
        if mitigation_label == unique_label:
            labels[label2id[unique_label]] = 1.
            # if mitigation_label == "Principal climate mitigation objective":
            #     labels[label2id["Significant climate mitigation objective"]] = 1.
    example['labels'] = labels
    return example


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def inference(model, text):
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True).to(DEVICE)
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = sigmoid(logits)
    predicted_classes = (predicted_confidences > 0.5)

    return predicted_classes, predicted_confidences


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(predictions, labels):
   predictions = np.array(predictions)
   labels = np.array(labels)
   predictions = (predictions > 0.5).astype(int)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int))


def run_inference(example):
    _, predictions = inference(MODEL, example['text'])
    labels = np.array(example['labels'])
    for unique_label in unique_labels:
        id = label2id[unique_label]
        pred_varname = 'Predicted {}'.format(unique_label)
        label_varname = 'Label {}'.format(unique_label)
        example[pred_varname] = predictions[id]
        example[label_varname] = labels[id]
    return example


dataset = load_dataset("alex-miller/curated-iati-climate", split="test")
dataset = dataset.map(preprocess_function, remove_columns=['adaptation_label', 'mitigation_label'])
dataset = dataset.map(run_inference)

for unique_label in unique_labels:
    print(unique_label)
    pred_varname = 'Predicted {}'.format(unique_label)
    label_varname = 'Label {}'.format(unique_label)
    print(compute_metrics(dataset[pred_varname], dataset[label_varname]))