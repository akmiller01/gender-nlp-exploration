import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import click
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm
from huggingface_hub import login
import re
from collections import Counter


load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)
HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)


MODEL = "gpt-3.5-turbo-0125"
MULTIPLIER = 10


disability_keywords = [
    'accessible',
    'accessibility',
    'albino',
    'albinism',
    'autism',
    'autistic',
    'blind',
    'blindness',
    'chronic',
    'deaf',
    'deafness',
    'déficience',
    'deformity',
    'deformities',
    'difficult',
    'difficulty',
    'difficulties',
    'dignified',
    'disabilitazione',
    'disabilities',
    'disability',
    'disable',
    'disabled',
    'discrimination',
    'eye',
    'eyes',
    'handicap',
    'handicapped',
    'handicapés',
    'handicapées',
    'hearing',
    'helpage',
    'impaired',
    'impairment',
    'impairments',
    'inclusion',
    'inclusive',
    'marginalization',
    'marginalized',
    'mental',
    'parkinson',
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
global DISABILITY_REGEX
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


def warn_user_about_tokens(tokenizer, text):
    token_cost = 0.5
    cost_per = 1000000
    token_count = len(tokenizer.encode(text))
    return click.confirm(
        "This will use at least {} tokens and cost at least ${} to run. Do you want to continue?".format(
        token_count, round((token_count / cost_per) * token_cost, 4)
    )
    , default=False)


def filter_keyword_match(example):
    clean_text = remove_string_special_characters(example['text'])
    return DISABILITY_REGEX.search(clean_text) is not None


if __name__ == '__main__':

    dataset = load_dataset("devinitorg/iati-policy-markers", split="train")

    dataset = dataset.filter(lambda example: example["disability_sig"] in [1, 2])
    dataset = dataset.filter(lambda example: example["text"] != "" and example["text"] is not None and len(example["text"]) > 10)
    dataset = dataset.filter(lambda example: filter_keyword_match(example))

    def relabel(example):
        if example['disability_sig'] == 1:
            label = "Significant"
        else:
            label = "Principal"
        example['label'] = label
        return example

    dataset = dataset.map(relabel, num_proc=8)

    cols_to_remove = dataset.column_names
    cols_to_remove.remove("text")
    cols_to_remove.remove("label")
    dataset = dataset.remove_columns(cols_to_remove)

    count = Counter()
    count.update(dataset['label'])
    print(count)

    dataset = dataset.add_column("class_labels", dataset['label'])

    dataset = dataset.class_encode_column('class_labels').train_test_split(
        test_size=0.8,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )

    dataset = dataset.remove_columns(["class_labels"])
    dataset_train = dataset['train']

    # format (Symantic description of label, extra instructions)
    semantic_label_mapping = {
        'Significant': ('Significant disability objective', 'Significant means that addressing disability is an important and deliberate objective, but not the principal reason for undertaking the project/programme, often explained as disability being mainstreamed in the project/programme'),
        'Principal': ('Principal disability objective', 'Principal means that addressing disability is the main objective of the project/programme and is fundamental is its design and expected results. The project/programme would not have been undertaken without this objective'),
    }
    system_prompt_format = "Below is a record from a database of development and humanitarian assistance. I need your help to create synthetic data to train a classifier network. Could you please write {} synthetic records based on the example, separated by new lines, that mirrors it in length, content, vocabulary, language, and theme? The synthetic record should reflect the theme we are trying to classify, which is '{}'. {}. Please only write the synthetic data and no additional text."

    def apply_system_prompts(example):
        semantic_label, extra_instructions = semantic_label_mapping[example["label"]]
        example["system_prompt"] = system_prompt_format.format(MULTIPLIER, semantic_label, extra_instructions)
        return example
    
    dataset_train = dataset_train.map(apply_system_prompts, num_proc=8)
    all_prompts = " ".join(dataset_train["system_prompt"])
    dataset_texts = " ".join(dataset_train["text"] * (MULTIPLIER + 1))
    all_text = all_prompts + dataset_texts
    tokenizer = tiktoken.encoding_for_model(MODEL)

    if warn_user_about_tokens(tokenizer, text=all_text) == True:
        synthetic_labels = list()
        synthetic_texts = list()
        for i, user_prompt in tqdm(enumerate(dataset_train["text"]), total=dataset_train.num_rows):
            system_prompt = dataset_train["system_prompt"][i]
            label = dataset_train["labels"][i]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                for synthetic_text in response.choices[0].message.content.split("\n"):
                    if synthetic_text != '':
                        synthetic_texts.append(synthetic_text)
                        synthetic_labels.append(label)
            except:
                print("Error fetching result {} from OpenAI.".format(i))

        synthetic_dataset = Dataset.from_dict({
            'text': synthetic_texts,
            'labels': synthetic_labels
        })
        dataset['train'] = concatenate_datasets([dataset['train'], synthetic_dataset])
        dataset.push_to_hub("alex-miller/iati-disability-synthetic")
        synthetic_dataset.to_csv("./large_data/iati-disability-synthetic.csv")
