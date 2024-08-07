# ! pip install datasets evaluate transformers accelerate huggingface_hub --quiet

# from huggingface_hub import login

# login()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import torch
import evaluate
import numpy as np

from bert_cls_pooled_normalized_model import BertForSequenceClassificationPooledNormalized


card = "alex-miller/ODABert"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = load_dataset("alex-miller/curated-iati-gender-equality")

unique_labels = [
    "Significant gender equality objective",
    "Principal gender equality objective"
]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

def preprocess_function(example):
    labels = [0. for i in range(len(unique_labels))]
    for label in unique_labels:
        if example['label'] == label:
            labels[label2id[label]] = 1.

    example = tokenizer(example['text'], truncation=True)
    example['labels'] = labels
    return example

dataset = dataset.map(preprocess_function, remove_columns=['text', 'label'])

weight_list = list()
total_rows = dataset['train'].num_rows + dataset['test'].num_rows
print("Weights:")
for label in unique_labels:
    label_idx = label2id[label]
    positive_filtered_dataset = dataset.filter(lambda example: example['labels'][label_idx] == 1.)
    negative_filtered_dataset = dataset.filter(lambda example: example['labels'][label_idx] == 0.)
    pos_label_rows = positive_filtered_dataset['train'].num_rows + positive_filtered_dataset['test'].num_rows
    neg_label_rows = negative_filtered_dataset['train'].num_rows + negative_filtered_dataset['test'].num_rows
    label_weight = neg_label_rows / pos_label_rows
    weight_list.append(label_weight)
    print("{}: {}".format(label, label_weight))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
weights = torch.tensor(weight_list)
weights = weights.to(device)

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


model = BertForSequenceClassificationPooledNormalized.from_pretrained(
    card,
    num_labels=len(id2label.keys()), 
    id2label=id2label,
    label2id=label2id, 
    problem_type="multi_label_classification"
)
model.class_weights = weights

training_args = TrainingArguments(
    'curated-gender-equality-weighted-normalized',
    learning_rate=1e-6, # This can be tweaked depending on how loss progresses
    per_device_train_batch_size=24, # These should be tweaked to match GPU VRAM
    per_device_eval_batch_size=24,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=True,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()