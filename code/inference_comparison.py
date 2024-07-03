from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets

from bert_cls_pooled_normalized_model import BertForSequenceClassificationPooledNormalized


global TOKENIZER
global DEVICE
global MODEL
global MODEL_NORM
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/curated-gender-equality-weighted', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/curated-gender-equality-weighted")
MODEL = MODEL.to(DEVICE)
MODEL_NORM = BertForSequenceClassificationPooledNormalized.from_pretrained("alex-miller/curated-gender-equality-weighted-normalized")
MODEL_NORM = MODEL_NORM.to(DEVICE)


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

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True).to(DEVICE)
    model_pred, model_conf = inference(MODEL, inputs)
    example['Gender equality - significant objective predicted'] = model_pred[0]
    example['Gender equality - significant objective confidence'] = model_conf[0]
    example['Gender equality - principal objective predicted'] = model_pred[1]
    example['Gender equality - principal objective confidence'] = model_conf[1]
    return example

def main():
    samples = [
        ("No gender equality objective - Real test sample", "Integrated rural and sustainable development focussing food security and biodiversity, Visayas and Luzon Region Förderung einer diversifizierten, nachhaltigen Landwirtschaft für Ernährungssicherung der Kleinbauernfamilien der Region Visayas und Luzon Integrated rural and sustainable development focussing food security and biodiversity, Visayas and Luzon Region Förderung einer diversifizierten, nachhaltigen Landwirtschaft für Ernährungssicherung der Kleinbauernfamilien der Region Visayas und Luzon Integrated rural and sustainable development focussing food security and biodiversity, Visayas and Luzon Region Förderung einer diversifizierten, nachhaltigen Landwirtschaft für Ernährungssicherung der Kleinbauernfamilien der Region Visayas und Luzon"),
        ("Significant gender equality objective - Real test sample", "EPI IN PHC By 2027, the capacities of the primary healthcare (PHC) system are strengthened to be more resilient and effectively manage and deliver life-saving vaccines, stimulating demand for quality immunization services and strengthening integrated PHC provision, especially amongst marginalised communities, both in development and humanitarian situations., which contributes to Health And Development In Early Childhood And Adolescence, Immunization Services As Part Of Primary Health Care, Strengthening Primary Health Care And High-Impact Health Interventions. UNICEF aims to achieve this through Advocacy And Communications, Community Engagement, Social And Behaviour Change, Digital Transformation, Operational Support To Programme Delivery, Partnerships And Engagement: Public And Private, Risk-Informed Humanitarian And Development Nexus Programming, Systems Strengthening To Leave No One Behind. This contributes to the following Country Programme result: By 2027, more women, newborns, children, and adolescents, especially those from marginalized and vulnerable groups, utilize quality, comprehensive, gender- and shock-responsive healthcare and HIV services, and benefit from nurturing practices and essential supplies."),
        ("Principal gender equality objective - Real test sample", "UN Women communication and visibility are enhanced to effectively advocate for gender equality and the empowerment of women UN Women communication and visibility are enhanced to effectively advocate for gender equality and the empowerment of women"),
        ("No gender equality objective - Synthetic short sample", "Health"),
        ("Principal equality objective - Synthetic short sample", "Women"),
    ]
    for sample_label, sample_text in samples:
        print("\n" + sample_label)
        inputs = TOKENIZER(sample_text, return_tensors="pt", truncation=True).to(DEVICE)
        _, model_conf = inference(MODEL, inputs)
        _, model_norm_conf = inference(MODEL_NORM, inputs)
        print("Unnormalized:")
        print(model_conf)
        print("Normalized:")
        print(model_norm_conf)


if __name__ == '__main__':
    main()

