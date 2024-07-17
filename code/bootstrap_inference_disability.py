from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
import re


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-disability-multi-classifier-weighted")
MODEL = MODEL.to(DEVICE)


disability_keywords = [
    'accessible',
    'accessibility',
    'albino', # new
    'albinism', # new
    'autism', # new
    'autistic', # new
    'blind',
    'blindness',
    'chronic', # new
    'deaf',
    'deafness',
    'déficience', # new
    'deformity', # new
    'deformities', # new
    'difficult', # new
    'difficulty', # new
    'difficulties', # new
    'dignified',
    'disabilitazione',
    'disabilities',
    'disability',
    'disable', # new
    'disabled',
    'discrimination',
    'eye', # new
    'eyes', # new
    'handicap',
    'handicapped',
    'handicapés', # new
    'handicapées', # new
    'hearing',
    'helpage', # new
    'impaired', # new
    'impairment', # new
    'impairments', # new
    'inclusion',
    'inclusive', # new
    'marginalization',
    'marginalized',
    'mental',
    'parkinson', # new
    'physical',
    'precondition',
    'seeing',
    'sight',
    'stigma',
    'stigmatisation',
    'stigmatization',
    'taboo',
    'vulnerability',
    'vulnerable',
    'wheelchair'
]

disability_regex_string = '|'.join([r'\b%s\b' % word for word in disability_keywords])
DISABILITY_REGEX = re.compile(disability_regex_string, re.I)


def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub(r'[^\w\s]', ' ', s)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped.lower()


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
    clean_text = remove_string_special_characters(text)
    example['Disability keyword match'] = DISABILITY_REGEX.search(clean_text) is not None

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True).to(DEVICE)
    model_pred, model_conf = inference(MODEL, inputs)
    example['Disability - significant objective predicted'] = model_pred[0]
    example['Disability - significant objective confidence'] = model_conf[0]
    example['Disability - principal objective predicted'] = model_pred[1]
    example['Disability - principal objective confidence'] = model_conf[1]
    return example

def main():
    dataset = load_dataset("devinitorg/iati-policy-markers", split="train")

    dataset = dataset.filter(lambda example: example["disability"])
    dataset = dataset.filter(lambda example: example["disability_sig"] in [0, 1, 2])
    dataset = dataset.filter(lambda example: example["text"] != "" and example["text"] is not None and len(example["text"]) > 10)
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("text")
    cols_to_remove.remove("disability_sig")
    dataset = dataset.remove_columns(cols_to_remove)

    # De-duplicate
    df = pd.DataFrame(dataset)
    print(df.shape)
    df = df.drop_duplicates(subset=['text'])
    print(df.shape)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    dataset = dataset.map(map_columns)
    dataset = pd.DataFrame(dataset)
    dataset.to_csv('large_data/iati_disability_predictions.csv', index=False)


if __name__ == '__main__':
    main()

