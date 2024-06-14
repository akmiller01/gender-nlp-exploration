from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
import math
import re


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-gender-multi-classifier-weighted-minitest")
MODEL = MODEL.to(DEVICE)

gender_keywords = [
    'abuse',
    'abused',
    'adolescent',
    'adolescents',
    'aids',
    'antirotavirus',
    'bodies',
    'body',
    'boys',
    'care',
    'cedaw', # Convention on the Elimination of All Forms of Discrimination Against Women
    'child',
    'childhood',
    'children',
    'daughter',
    'equality',
    'exploitation',
    'exploited',
    'familien',
    'families',
    'family',
    'female',
    'feminism',
    'feminist',
    'femmes',
    'force',
    'forced',
    'frauenrechten',
    'garçons',
    'gbv', # gender-based violence
    'gender',
    'genital',
    'genitals',
    'girl',
    'girls',
    'haemophilius',
    'haemophilus',
    'hiv',
    'inequality',
    'jeunes',
    'könsbaserat',
    'masculine',
    'masculinity',
    'maternal',
    'menstrual',
    'menstruation',
    'misogyny',
    'mother',
    'mothers',
    'normative',
    'pad',
    'pads',
    'patriarchy',
    'period',
    'periods',
    'rape',
    'raped',
    'reproduce',
    'reproductive',
    'sanitary',
    'sex',
    'sexism',
    'sexospécifique',
    'sexual',
    'sexueller',
    'son',
    'stem', # Science, technology, engineering and maths?
    'tampon',
    'tampons',
    'tgnp', # Tanzania Gender Networking Programme
    'ungdomar',
    'unwomen',
    'uterus',
    'utérus',
    'vaw', # violence against women
    'violence',
    'violent',
    'woman',
    'women',
    'wps', # women peace security
    'young',
    'youth',
    'youths'
]

gender_regex_string = '|'.join([r'\b%s\b' % word for word in gender_keywords])
GENDER_REGEX = re.compile(gender_regex_string, re.I)


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


def chunk_by_tokens(input_text, model_max_size=512):
    chunks = list()
    tokens = TOKENIZER.encode(input_text)
    token_length = len(tokens)
    if token_length <= model_max_size:
        return [input_text]
    desired_number_of_chunks = math.ceil(token_length / model_max_size)
    calculated_chunk_size = math.ceil(token_length / desired_number_of_chunks)
    for i in range(0, token_length, calculated_chunk_size):
        chunks.append(TOKENIZER.decode(tokens[i:i + calculated_chunk_size]))
    return chunks


def inference(model, inputs):
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = sigmoid(logits)
    predicted_classes = (predicted_confidences > 0.5)

    return predicted_classes, predicted_confidences

def map_columns(example):
    text = example['text']
    clean_text = remove_string_special_characters(text)

    predictions = {
        "Gender equality - significant objective": [False, 0],
        "Gender equality - principal objective": [False, 0],
    }
    keyword_match = False

    if text is not None:
        keyword_match = GENDER_REGEX.search(clean_text) is not None
        text_chunks = chunk_by_tokens(text)
        for text_chunk in text_chunks:
            inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
            model_pred, model_conf = inference(MODEL, inputs)
            predictions['Gender equality - significant objective'][0] = predictions['Gender equality - significant objective'][0] or model_pred[0]
            predictions['Gender equality - significant objective'][1] = max(predictions['Gender equality - significant objective'][1], model_conf[0])
            predictions['Gender equality - principal objective'][0] = predictions['Gender equality - principal objective'][0] or model_pred[1]
            predictions['Gender equality - principal objective'][1] = max(predictions['Gender equality - principal objective'][1], model_conf[1])
        
    example['Gender equality - significant objective predicted'] = predictions['Gender equality - significant objective'][0]
    example['Gender equality - significant objective confidence'] = predictions['Gender equality - significant objective'][1]
    example['Gender equality - principal objective predicted'] = predictions['Gender equality - principal objective'][0]
    example['Gender equality - principal objective confidence'] = predictions['Gender equality - principal objective'][1]
    example['Gender keyword match'] = keyword_match
    return example

def main():
    text_cols = ['text']
    dataset = load_dataset("devinitorg/iati-policy-markers", split="train")

    dataset = dataset.filter(lambda example: example["gender_equality"])
    dataset = dataset.filter(lambda example: example["gender_equality_sig"] in [0, 1, 2])
    dataset = dataset.filter(lambda example: example["text"] != "" and example["text"] is not None and len(example["text"]) > 10)
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("text")
    cols_to_remove.remove("gender_equality_sig")
    dataset = dataset.remove_columns(cols_to_remove)

    # De-duplicate
    df = pd.DataFrame(dataset)
    print(df.shape)
    df = df.drop_duplicates(subset=['text'])
    print(df.shape)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    dataset_text = dataset[text_cols]
    dataset_text = Dataset.from_pandas(dataset_text)
    dataset_text = dataset_text.map(map_columns, remove_columns=text_cols)
    dataset_text = pd.DataFrame(dataset_text)
    dataset = pd.concat([dataset.reset_index(drop=True), dataset_text.reset_index(drop=True)], axis=1)
    dataset.to_csv('large_data/iati_predictions.csv', index=False)


if __name__ == '__main__':
    main()

