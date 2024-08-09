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
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-climate-multi-classifier-weighted2")
MODEL = MODEL.to(DEVICE)


climate_keywords = [
    'adapt',
    'adaptation',
    'adaptative',
    'adaption',
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
    'air',
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
    'catastrophic',
    'catástrofe',
    'catástrofes',
    'ccnucc',
    'cement',
    'cgiar',
    'charcoal',
    'chemin de fer',
    'chemins de fer',
    'clean',
    'cleaner',
    'clim',
    'climat',
    'climate',
    'climateshot',
    'climatic',
    'climatico',
    'climatique',
    'climatiques',
    'climático',
    'cng',
    'coal',
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
    'energetica',
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
    'gef',
    'geotermia',
    'geothermal',
    'ghg',
    'global warming',
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
    'inun daciones',
    'inundacion',
    'inundaciones',
    'inundación',
    'iucn',
    'klimatanpassning',
    'kyoto',
    "l'électricité",
    'land',
    'land',
    'landdegradatie',
    'lands',
    'lcf',
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
    'montreal',
    'mw',
    'météorologique',
    'météorologiques',
    'nationally determined contribution',
    'nationally determined contributions',
    'natural',
    'nature',
    'naturelles',
    'ndc',
    'ndcs',
    'ocean',
    'odnawialne',
    'odpadów',
    'organic',
    'ozone',
    'paris',
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
    'risk insurance',
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
    'sequia',
    'sequias',
    'sequía',
    'sequía',
    'sequías',
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

climate_regex_string = '|'.join([r'\b%s\b' % word for word in climate_keywords])
CLIMATE_REGEX = re.compile(climate_regex_string, re.I)


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
    example['Climate keyword match'] = CLIMATE_REGEX.search(clean_text) is not None

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True).to(DEVICE)
    _, climate_model_conf = inference(MODEL, inputs)
    example['Climate adaptation - significant objective'] = climate_model_conf[0]
    example['Climate adaptation - principal objective'] = climate_model_conf[1]
    example['Climate mitigation - significant objective'] = climate_model_conf[2]
    example['Climate mitigation - principal objective'] = climate_model_conf[3]
    return example

def main():
    dataset = load_dataset("devinitorg/iati-policy-markers", split="train")

    dataset = dataset.filter(lambda example: example["climate_adaptation"] or example["climate_mitigation"])
    dataset = dataset.filter(lambda example: example["climate_adaptation_sig"] in [0, 1, 2] or example["climate_mitigation_sig"] in [0, 1, 2])
    dataset = dataset.filter(lambda example: example["text"] != "" and example["text"] is not None and len(example["text"]) > 10)
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("text")
    cols_to_remove.remove("climate_adaptation_sig")
    cols_to_remove.remove("climate_mitigation_sig")
    dataset = dataset.remove_columns(cols_to_remove)

    # De-duplicate
    df = pd.DataFrame(dataset)
    print(df.shape)
    df = df.drop_duplicates(subset=['text'])
    print(df.shape)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    dataset = dataset.map(map_columns)
    dataset = pd.DataFrame(dataset)
    dataset.to_csv('large_data/iati_climate_predictions.csv', index=False)


if __name__ == '__main__':
    main()

