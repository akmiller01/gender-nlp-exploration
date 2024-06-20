from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from bert_unpooled_model import BertForSequenceClassificationUnpooled


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/curated-gender-equality-weighted-unpooled', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = BertForSequenceClassificationUnpooled.from_pretrained("alex-miller/curated-gender-equality-weighted-unpooled")
MODEL = MODEL.to(DEVICE)


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def inference(model, inputs):
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = sigmoid(logits)
    predicted_classes = (predicted_confidences > 0.5)

    return predicted_classes, predicted_confidences


def map_columns(example):
    text = example['text']

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding='max_length').to(DEVICE)
    model_pred, model_conf = inference(MODEL, inputs)
    example['Gender equality - significant objective predicted'] = model_pred[0]
    example['Gender equality - significant objective confidence'] = model_conf[0]
    example['Gender equality - principal objective predicted'] = model_pred[1]
    example['Gender equality - principal objective confidence'] = model_conf[1]
    return example

def main():
    dataset = load_dataset("alex-miller/curated-iati-gender-equality", split="test")
    dataset = dataset.map(map_columns)
    dataset = pd.DataFrame(dataset)
    dataset.to_csv('large_data/iati_predictions3.csv', index=False)


if __name__ == '__main__':
    main()

