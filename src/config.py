

POSTS_PATH = "data/complete_data/posts.csv"
FACT_CHECKS_PATH = "data/complete_data/fact_checks.csv"
TASKS_PATH = "data/splits/tasks_no_gs_overlap.json"
GS_PATH = "data/complete_data/pairs.csv"
PHASE1_TASKS_PATH = "data/complete_data/tasks.json"
TEST_PHASE_TASKS_PATH = "data/splits/splits_test.json"

OUTPUT_PATH = "out"
LANGS = ['eng', 'fra', 'deu', 'por', 'spa', 'tha',  'msa', 'ara']
TEST_PHASE_LANGS = ['eng', 'fra', 'deu', 'por', 'spa', 'tha',  'msa', 'ara', 'pol', 'tur']

MINILM6_EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
MINILM12_MULTILINGUAL_EMBED = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MINILM12_ALL = 'sentence-transformers/all-MiniLM-L12-v2'

MINILM6_CROSS = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
MINILM12_CROSS = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

ROBERTA_CROSS = "cross-encoder/stsb-roberta-large"

E5ENCODER = "intfloat/multilingual-e5-large"

MPNET_ENCODER = "sentence-transformers/all-mpnet-base-v2" 

SPACY_MODELS = {
    'eng': 'en_core_web_sm',
    'fra': 'fr_core_news_sm',
    'deu': 'de_core_news_sm',
    'por': 'pt_core_news_sm',
    'spa': 'es_core_news_sm',
    'tha':  None,
    'msa':  None,
    'ara':  None
}

LANG_COUNTRY = {
    'spa': 'Spain',
    'am': 'Ethiopia',
    'tgl': 'Philippines',
    'urd': 'Pakistan',
    'cat': 'Catalonia (Spain)',
    'mg': 'Madagascar',
    'so': 'Somalia',
    'ln': 'Democratic Republic of the Congo',
    'co': 'France (Corsica)',
    'ay': 'Bolivia',
    'lg': 'Uganda',
    'ces': 'Czech Republic',
    'xh': 'South Africa',
    'fy': 'Netherlands (Friesland)',
    'tha': 'Thailand',
    'jpn': 'Japan',
    'slk': 'Slovakia',
    'ron': 'Romania',
    'mo': 'Moldova',
    'mya': 'Myanmar',
    'sl': 'Slovenia',
    'bul': 'Bulgaria',
    'ti': 'Eritrea',
    'haw': 'United States (Hawaii)',
    'sw': 'Tanzania',
    'nso': 'South Africa',
    'af': 'South Africa',
    'deu': 'Germany',
    'la': 'Vatican City',
    'gl': 'Spain (Galicia)',
    'jw': 'Indonesia',
    'fra': 'France',
    'nor': 'Norway',
    'khm': 'Cambodia',
    'lv': 'Latvia',
    'ha': 'Nigeria',
    'kri': 'Sierra Leone',
    'zu': 'South Africa',
    'ita': 'Italy',
    'gu': 'India (Gujarat)',
    'hmn': 'China (Hmong community)',
    'ig': 'Nigeria',
    'eu': 'Spain (Basque Country)',
    'mi': 'New Zealand',
    'yo': 'Nigeria',
    'nld': 'Netherlands',
    'msa': 'Malaysia',
    'ht': 'Haiti',
    'hi-Latn': 'India',
    'ara': 'Saudi Arabia',
    'tel': 'India (Telangana)',
    'kn': 'India (Karnataka)',
    'mr': 'India (Maharashtra)',
    'hin': 'India',
    'su': 'Indonesia',
    'fin': 'Finland',
    'qu': 'Peru',
    'hbs': 'Serbia',
    'ell': 'Greece',
    'uz': 'Uzbekistan',
    'sin': 'Sri Lanka',
    'ben': 'Bangladesh',
    'tur': 'Turkey',
    'ts': 'South Africa',
    'ru-Latn': 'Russia',
    'hun': 'Hungary',
    'st': 'Lesotho',
    'ny': 'Malawi',
    'eng': 'United Kingdom',
    'ceb': 'Philippines',
    'kor': 'South Korea',
    'rus': 'Russia',
    'lb': 'Luxembourg',
    'dan': 'Denmark',
    'gd': 'Scotland (United Kingdom)',
    'zho': 'China',
    'pol': 'Poland',
    'sv': 'Sweden',
    'tam': 'India (Tamil Nadu)',
    'fil': 'Philippines',
    'et': 'Estonia',
    'por': 'Portugal'
}

