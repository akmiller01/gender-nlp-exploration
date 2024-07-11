from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import math
import re


global TOKENIZER
global DEVICE
global GENDER_MODEL
global CLIMATE_MODEL
global DISABILITY_MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GENDER_MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/curated-gender-equality-weighted")
GENDER_MODEL = GENDER_MODEL.to(DEVICE)
CLIMATE_MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-climate-multi-classifier-weighted2")
CLIMATE_MODEL = CLIMATE_MODEL.to(DEVICE)
DISABILITY_MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-disability-multi-classifier-weighted")
DISABILITY_MODEL = DISABILITY_MODEL.to(DEVICE)

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

climate_keywords = [
    'adapt',
    'adaptation',
    'adaptative',
    'adaptive',
    'afforestation',
    'agri',
    'agricole',
    'agricoles',
    'agricultural',
    'agriculture',
    'agro',
    'agroecological',
    'agroecology',
    'agroforestry',
    'agrícola',
    'ambiental',
    'anthracnose',
    'aquaponics',
    'baterias',
    'batería',
    'batterie',
    'batteries',
    'battery',
    'bio',
    'biodiversity',
    'biodiversité',
    'bioenergy',
    'biomasa',
    'biomass',
    'biomasse',
    'bioremediation',
    'bosque',
    'bosques',
    'carbon',
    'carbone',
    'catastrophe',
    'catastrophes',
    'catástrofe',
    'catástrofes',
    'ccnucc',
    'cement',
    'cgiar',
    'charcoal',
    'chemin de fer',
    'chemins de fer',
    'clean',
    'climat',
    'climate',
    'climatic',
    'climatico',
    'climatique',
    'climatiques',
    'climático',
    'coastal',
    'coffee',
    'compost',
    'composting',
    'conservation',
    'consistently',
    'contribución determinada nacional',
    'contribution determinee nationale',
    'cook',
    'cooking',
    'coping',
    'crop',
    'crops',
    'cultivo',
    'cultivos',
    'culture',
    'cultures',
    "d'électricité",
    'decarbonization',
    'deforestación',
    'deforestation',
    'degradation',
    'depleted',
    'depletion',
    'desalination',
    'desert',
    'desertification',
    'desierto',
    'dessalement',
    'disaster',
    'disaster risk',
    'disasters',
    'diverse',
    'diversified',
    'diversifying',
    'drm',
    'drought',
    'drr',
    'dryland',
    'drylands',
    'durable',
    'durablement',
    'durables',
    'déchets',
    'déforestation',
    'désert',
    'early warning',
    'ecological',
    'ecologique',
    'ecologiques',
    'ecology',
    'ecosistema',
    'ecosistemas',
    'ecosystem',
    'ecosystemen',
    'ecosystems',
    'efficiency',
    'electric',
    'electricity',
    'electricité',
    'electrification',
    'electrique',
    'elektronicznego',
    'elektrycznej',
    'elephant',
    'elephants',
    'emision',
    'emisiones',
    'emisión',
    'emission',
    'emissions',
    'energetique',
    'energi',
    'energia',
    'energie',
    'energies',
    'energy',
    'env',
    'environment',
    'environmental',
    'environnement',
    'environnementales',
    'environnementaux',
    'eolica',
    'eolienne',
    'exhaust',
    'eólica',
    'farm',
    'farmer',
    'farmers',
    'farms',
    'ferrocarril',
    'flloca',
    'flood',
    'flooding',
    'floods',
    'forest',
    'forestal',
    'forestiere',
    'forestry',
    'forests',
    'forêt',
    'forêts',
    'fotowoltaiczne',
    'fuel',
    'gas',
    'gases',
    'gaz',
    'gazów',
    'gcf',
    'geotermia',
    'geothermal',
    'ghg',
    'grazing',
    'green',
    'greenhouse',
    'greening',
    'grid',
    'grids',
    'grille',
    'géothermique',
    'harvest',
    'harvests',
    'hydro',
    'hydroelectric',
    'hydroelectriques',
    'hydropower',
    'hydroélectrique',
    'ifad',
    'interconnection',
    'interconnexion',
    'iucn', "l'électricité", 'land',
    'lcf',
    'land',
    'lands',
    'lignes',
    'limpio',
    'lines',
    'lowlands',
    'líneas',
    'mangrove',
    'mangroves',
    'marine',
    'meteorological',
    'meteorologique',
    'meteorologiques',
    'meteorology',
    'meteorológica',
    'meteorológicos',
    'mini-reseaux',
    'mitigación',
    'mitigated',
    'mitigating',
    'mitigation',
    'montane',
    'mw',
    'météorologique',
    'météorologiques',
    'nationally determined contribution',
    'nationally determined contributions',
    'natural',
    'nature',
    'naturelles',
    'ndc',
    'ocean',
    'odnawialne',
    'odpadów',
    'organic',
    'pastorales',
    'pays sec',
    'permaculture',
    'photovoltaic',
    'plant',
    'plantacji',
    'plantation',
    'plantations',
    'planting',
    'plants',
    'power',
    'preparedness',
    'propre',
    'pv',
    'railway',
    'railways',
    'rains',
    'recycle',
    'recycled',
    'recycling',
    'redd',
    'reforestation',
    'remediation',
    'remote sensing',
    'renewable',
    'renewables',
    'renouvelable',
    'renouvelables',
    'residuos',
    'resilience',
    'resilient',
    'restoration',
    'retrofitting',
    'reuse',
    'rice',
    'rio',
    'risk reduction',
    'rolnej',
    'rolniczej',
    'réseau',
    'réseaux',
    'résilience',
    'satellite',
    'sea',
    'seascape',
    'season',
    'sequestration',
    'soil',
    'solaire',
    'solaires',
    'solar',
    'solarization',
    'sols',
    'soneczn',
    'spalinowych',
    'species',
    'suelos',
    'sustainability',
    'sustainable',
    'sustainably',
    'sustentablemente',
    'sequía',
    'sécheresse',
    'terre',
    'terres',
    'territorial',
    'territory',
    'tierra',
    'tierras',
    'tolerant',
    'transmisión',
    'transmission',
    'tropical',
    'uicn',
    'unfccc',
    'upcycling',
    'vegetation',
    'verte',
    'vias ferreas',
    'waste',
    'water',
    'watershed',
    'weather',
    'weatherization',
    'wildlife',
    'wind',
    'windpower',
    'zielona',
    'zone',
    'zones',
    'écologique',
    'écologiques',
    'écosystème',
    'écosystèmes',
    'écosystémiques',
    'électrique',
    'émission',
    'émissions',
    'énergie',
    'énergétique',
    'éolienne'
]

disability_keywords = [
    'accessible',
    'accessibility',
    'blind',
    'blindness',
    'deaf',
    'deafness',
    'dignified',
    'disabilitazione',
    'disabilities',
    'disability',
    'disabled',
    'discrimination',
    'handicap',
    'handicapped',
    'hearing',
    'inclusion',
    'marginalization',
    'marginalized',
    'mental',
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

gender_regex_string = '|'.join([r'\b%s\b' % word for word in gender_keywords])
GENDER_REGEX = re.compile(gender_regex_string, re.I)

climate_regex_string = '|'.join([r'\b%s\b' % word for word in climate_keywords])
CLIMATE_REGEX = re.compile(climate_regex_string, re.I)

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
    screened_digits = [0, 1, 2]
    gender_screened = example['gender'] in screened_digits
    climate_screened = (example['climate_adaptation'] in screened_digits) or (example['climate_mitigation'] in screened_digits)
    disability_screened = example['disability'] in screened_digits
    textual_data_list = [
        example['project_title'],
        example['short_description'],
        example['long_description']
    ]
    textual_data_list = [str(textual_data) for textual_data in textual_data_list if textual_data is not None]
    text = remove_string_special_characters(" ".join(textual_data_list))

    predictions = {
        "Gender equality - significant objective": [
            example['gender'] == 1,
            1 if example['gender'] == 1 else 0
        ],
        "Gender equality - principal objective": [
            example['gender'] == 2,
            1 if example['gender'] == 2 else 0
        ],
        "Climate adaptation - significant objective": [
            example['climate_adaptation'] == 1,
            1 if example['climate_adaptation'] == 1 else 0
        ],
        "Climate adaptation - principal objective": [
            example['climate_adaptation'] == 2,
            1 if example['climate_adaptation'] == 2 else 0
        ],
        "Climate mitigation - significant objective": [
            example['climate_mitigation'] == 1,
            1 if example['climate_mitigation'] == 1 else 0
        ],
        "Climate mitigation - principal objective": [
            example['climate_mitigation'] == 2,
            1 if example['climate_mitigation'] == 2 else 0
        ],
        "Disability - significant objective": [
            example['disability'] == 1,
            1 if example['disability'] == 1 else 0
        ],
        "Disability - principal objective": [
            example['disability'] == 2,
            1 if example['disability'] == 2 else 0
        ],
    }
    gender_keyword_match = False
    climate_keyword_match = False
    disability_keyword_match = False

    if text is not None:
        gender_keyword_match = GENDER_REGEX.search(text) is not None
        climate_keyword_match = CLIMATE_REGEX.search(text) is not None
        disability_keyword_match = DISABILITY_REGEX.search(text) is not None
        text_chunks = chunk_by_tokens(text)
        for text_chunk in text_chunks:
            inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
            if not gender_screened:
                gender_model_pred, gender_model_conf = inference(GENDER_MODEL, inputs)
                predictions['Gender equality - significant objective'][0] = predictions['Gender equality - significant objective'][0] or gender_model_pred[0]
                predictions['Gender equality - significant objective'][1] = max(predictions['Gender equality - significant objective'][1], gender_model_conf[0])
                predictions['Gender equality - principal objective'][0] = predictions['Gender equality - principal objective'][0] or gender_model_pred[1]
                predictions['Gender equality - principal objective'][1] = max(predictions['Gender equality - principal objective'][1], gender_model_conf[1])
            if not climate_screened:
                climate_model_pred, climate_model_conf = inference(CLIMATE_MODEL, inputs)
                predictions['Climate adaptation - significant objective'][0] = predictions['Climate adaptation - significant objective'][0] or climate_model_pred[0]
                predictions['Climate adaptation - significant objective'][1] = max(predictions['Climate adaptation - significant objective'][1], climate_model_conf[0])
                predictions['Climate adaptation - principal objective'][0] = predictions['Climate adaptation - principal objective'][0] or climate_model_pred[1]
                predictions['Climate adaptation - principal objective'][1] = max(predictions['Climate adaptation - principal objective'][1], climate_model_conf[1])
                predictions['Climate mitigation - significant objective'][0] = predictions['Climate mitigation - significant objective'][0] or climate_model_pred[2]
                predictions['Climate mitigation - significant objective'][1] = max(predictions['Climate mitigation - significant objective'][1], climate_model_conf[2])
                predictions['Climate mitigation - principal objective'][0] = predictions['Climate mitigation - principal objective'][0] or climate_model_pred[3]
                predictions['Climate mitigation - principal objective'][1] = max(predictions['Climate mitigation - principal objective'][1], climate_model_conf[3])
            if not disability_screened:
                disability_model_pred, disability_model_conf = inference(DISABILITY_MODEL, inputs)
                predictions['Disability - significant objective'][0] = predictions['Disability - significant objective'][0] or disability_model_pred[0]
                predictions['Disability - significant objective'][1] = max(predictions['Disability - significant objective'][1], disability_model_conf[0])
                predictions['Disability - principal objective'][0] = predictions['Disability - principal objective'][0] or disability_model_pred[1]
                predictions['Disability - principal objective'][1] = max(predictions['Disability - principal objective'][1], disability_model_conf[1])


    example['Gender equality - significant objective predicted'] = predictions['Gender equality - significant objective'][0]
    example['Gender equality - significant objective confidence'] = predictions['Gender equality - significant objective'][1]
    example['Gender equality - principal objective predicted'] = predictions['Gender equality - principal objective'][0]
    example['Gender equality - principal objective confidence'] = predictions['Gender equality - principal objective'][1]
    example['Gender keyword match'] = gender_keyword_match
    example['Climate adaptation - significant objective predicted'] = predictions['Climate adaptation - significant objective'][0]
    example['Climate adaptation - significant objective confidence'] = predictions['Climate adaptation - significant objective'][1]
    example['Climate adaptation - principal objective predicted'] = predictions['Climate adaptation - principal objective'][0]
    example['Climate adaptation - principal objective confidence'] = predictions['Climate adaptation - principal objective'][1]
    example['Climate mitigation - significant objective predicted'] = predictions['Climate mitigation - significant objective'][0]
    example['Climate mitigation - significant objective confidence'] = predictions['Climate mitigation - significant objective'][1]
    example['Climate mitigation - principal objective predicted'] = predictions['Climate mitigation - principal objective'][0]
    example['Climate mitigation - principal objective confidence'] = predictions['Climate mitigation - principal objective'][1]
    example['Climate keyword match'] = climate_keyword_match
    example['Disability - significant objective predicted'] = predictions['Disability - significant objective'][0]
    example['Disability - significant objective confidence'] = predictions['Disability - significant objective'][1]
    example['Disability - principal objective predicted'] = predictions['Disability - principal objective'][0]
    example['Disability - principal objective confidence'] = predictions['Disability - principal objective'][1]
    example['Disability keyword match'] = disability_keyword_match
    return example

def main():
    text_cols = ['project_title', 'short_description', 'long_description', 'gender', 'climate_adaptation', 'climate_mitigation', 'disability']
    dataset = pd.read_csv("large_data/crs_for_gender_climate_disability.csv")
    dataset_screened = dataset[
        dataset['gender'].isin([0, 1, 2]) &
        (dataset['climate_adaptation'].isin([0, 1, 2]) | dataset['climate_mitigation'].isin([0, 1, 2])) &
        dataset['disability'].isin([0, 1, 2])
    ]
    dataset_unscreened = dataset[
        dataset['gender'].isnull() |
        (dataset['climate_adaptation'].isnull() & dataset['climate_mitigation'].isnull()) |
        dataset['disability'].isnull()
    ]
    dataset_text = dataset_unscreened[text_cols]
    dataset_text = Dataset.from_pandas(dataset_text)
    dataset_text = dataset_text.map(map_columns, remove_columns=text_cols)
    dataset_text = pd.DataFrame(dataset_text)
    dataset_unscreened = pd.concat([dataset_unscreened.reset_index(drop=True), dataset_text.reset_index(drop=True)], axis=1)
    dataset = pd.concat([dataset_screened, dataset_unscreened])
    dataset.to_csv('large_data/crs_for_gender_climate_disability_predictions.csv', index=False)


if __name__ == '__main__':
    main()

