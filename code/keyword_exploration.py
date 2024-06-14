from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd


def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub(r'[^\w\s]', ' ', s)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
            return stripped.lower()
    
def remove_stop_words(s):
    stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish')) | set(stopwords.words('french')) | set(stopwords.words('german'))
    try:
        return ' '.join([word for word in nltk.word_tokenize(s) if word not in stop_words])
    except TypeError:
         return ' '
    

def features(vectorizer, result):
    feature_array = np.array(vectorizer.get_feature_names_out())
    flat_values = result.toarray().flatten()
    zero_indices = np.where(flat_values == 0)[0]
    tfidf_sorting = np.argsort(flat_values)[::-1]
    masked_tfidf_sorting = np.delete(tfidf_sorting, zero_indices, axis=0)

    return feature_array[masked_tfidf_sorting].tolist()


def clean_text(example):
    example['text'] = remove_stop_words(remove_string_special_characters(example['text']))
    return example


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
dataset = dataset.map(clean_text, num_proc=8)

not_gender = dataset.filter(lambda example: example['gender_equality_sig'] == 0)
sig_gender = dataset.filter(lambda example: example['gender_equality_sig'] == 1)
prin_gender = dataset.filter(lambda example: example['gender_equality_sig'] == 2)

vectorizer = TfidfVectorizer(ngram_range=(1, 1))
vectorizer.fit(dataset['text'])

not_gender_result = vectorizer.transform([" ".join(not_gender['text'])])
top_not_gender = features(vectorizer, not_gender_result)
top_not_gender = top_not_gender[:250]

sig_gender_result = vectorizer.transform([" ".join(sig_gender['text'])])
top_sig_gender = features(vectorizer, sig_gender_result)
top_sig_gender = [word for word in top_sig_gender if word not in top_not_gender]
top_sig_gender = top_sig_gender[:250]

prin_gender_result = vectorizer.transform([" ".join(prin_gender['text'])])
top_prin_gender = features(vectorizer, prin_gender_result)
top_prin_gender = [word for word in top_prin_gender if word not in top_not_gender]
top_prin_gender = [word for word in top_prin_gender if word not in top_sig_gender]
top_prin_gender = top_prin_gender[:250]


print("Significant:\n")
print(top_sig_gender)

print("Principal:\n")
print(top_prin_gender)