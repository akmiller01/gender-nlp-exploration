from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import os


def concat_labels(example):
    example['label'] = '{} {}'.format(example['adaptation_label'], example['mitigation_label'])
    return example

def main():
    dataset = load_dataset("csv", data_files="./large_data/curated_climate_training_data.csv", split="train")
    dataset = dataset.map(concat_labels)
    dataset = dataset.class_encode_column('label').train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        shuffle=True,
        seed=1337
    )
    dataset = dataset.remove_columns(['label'])
    dataset.push_to_hub("alex-miller/curated-iati-climate")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()
