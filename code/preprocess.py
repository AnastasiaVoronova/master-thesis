import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import tempfile
import requests
from tqdm import tqdm
import pymorphy3
from symspellpy import SymSpell, Verbosity
import nltk


def get_category_index(category):

    categories = {
    'нет конкретного ответа' : 0,
    '?' : 0,
    'ок' : 1,
    'work-life balance' : 2,
    'адекватные планы и количество метрик' : 3,
    'адекватные планы и кол-во метрик' : 3,
    'бесплатное питание' : 4,
    'бесплатная еда' : 4,
    'бюрократия' : 5,
    'взаимодействие' : 6,
    'взаимодействие ' : 6,
    'внерабочие активности' : 7,
    'график работы' : 8,
    'график' : 8,
    'дополнительные сотрудники' : 9,
    'идея по продукту' : 10,
    'идеи по продукту' : 10,
    'карьерный рост' : 11,
    'клиенты' : 12,
    'конкурсы' : 13,
    'культура обратной связи' : 14,
    'культура обратной связи ' : 14,
    'лояльность к сотрудникам' : 15,
    'льготы' : 16,
    'ль' : 16,
    'спортивный зал' : 16,
    'бассейн' : 16,
    'мерч' : 17,
    'нездоровая атмосфера' : 18,
    'обучение' : 19,
    'оплата труда' : 20,
    'оплата' : 20,
    'офисное пространство' : 21,
    'подарки на праздники' : 22,
    'подарки по праздникам' : 22,
    'подарки детям' : 22,
    'премии' : 23,
    'Премии' : 23,
    'процессы' : 24,
    'сложность работы' : 25,
    'техника/ит' : 26,
    'технологии/ит' : 26,
    'удаленная работа' : 27,
    'работа из заграницы' : 27,
    'работа из других стран' : 27,
    'оплата сверхурочного труда' : 28,
    'руководитель' : 29
    }

    try:
        return categories[category]
    except KeyError:
        return None
    
def get_category_name(index):

    categories_from_indices = {
    0 :  'нет конкретного ответа',
    1 :  'ок',
    2 :  'work-life balance',
    3 :  'адекватные планы и количество метрик',
    4 :  'бесплатное питание',
    5 :  'бюрократия',
    6 :  'взаимодействие',
    7 :  'внерабочие активности',
    8 :  'график работы',
    9 :  'дополнительные сотрудники',
    10 :  'идея по продукту',
    11 :  'карьерный рост',
    12 :  'клиенты',
    13 :  'конкурсы',
    14 :  'культура обратной связи',
    15 :  'лояльность к сотрудникам',
    16 :  'льготы',
    17 :  'мерч',
    18 :  'нездоровая атмосфера',
    19 :  'обучение',
    20 :  'оплата труда',
    21 :  'офисное пространство',
    22 :  'подарки на праздники',
    23 :  'премии',
    24 :  'процессы',
    25 :  'сложность работы',
    26 :  'техника/ит',
    27 : 'удаленная работа',
    28 : 'оплата сверхурочного труда',
    29 : 'руководитель'
    }

    try:
        return categories_from_indices[index]
    except KeyError:
        return None

def load_data(filename):
    if filename.endswith('xlsx'):
        df = pd.read_excel(filename)
    elif filename.endswith('csv'):
        df = pd.read_csv(filename)
    df1 = df.loc[:,['Score','A1','C1']].rename(columns={'Score':'S', 'A1':'A', 'C1':'C'})
    df2 = df.loc[:,['Score','A2','C2']].rename(columns={'Score':'S', 'A2':'A', 'C2':'C'})
    df = pd.concat([df1,df2]).dropna()
    df['Y'] = df['C'].apply(get_category_index)
    df['C'] = df['Y'].apply(get_category_name)
    return df

def delete_non_alpha(text):
    global stopwords
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zа-яё]', ' ', str(text))
    text = re.sub(r'\b\w\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_full_dict():
    url = "https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt"
    response = requests.get(url)
    response.raise_for_status()
    return set(response.text.splitlines())

def setup_symspell():

    url = "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/ru-100k.txt"
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as f:
        for line in response.text.splitlines():
            word, freq = line.split()
            f.write(f"{word}\t{freq}\n")
        temp_path = f.name

    sym_spell = SymSpell()
    sym_spell.load_dictionary(
        temp_path, 
        term_index=0, 
        count_index=1,
        separator="\t")
    
    return sym_spell

def get_stopwords():
    try:
        stopwords = nltk.corpus.stopwords.words("russian")
    except ModuleNotFoundError:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words("russian")
    antistopwords = ['не','нет','ни','ничего','без','никогда','нельзя','всегда','конечно','надо','хорошо','лучше','больше','более']
    for word in antistopwords:
        stopwords.remove(word)
    return set(stopwords)

def correct_text(words, full_dict, sym_spell):
    corrected = []
    for word in words:
        if not word in full_dict:
            suggestions = sym_spell.lookup(word, Verbosity.TOP)
            if suggestions:
                corrected.append(suggestions[0].term)
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    return corrected

def lemmatize(words, morph):
    lemmatized = []
    for word in words:
        lemmatized.append(morph.parse(word)[0].normal_form)
    return lemmatized

def delete_stop_words(words, stopwords):
    normal_words = []
    for word in words:
        if not word in stopwords:
            normal_words.append(word)
    return normal_words

def check_empty(words):
    if len(words) == 0:
        return None
    return words

def text_from_words(words):
    if words:
        return ' '.join(word for word in words)
    return None

def prepare_data(df, test_size=0.25,  syntax_correction=False, lemmatization=False, stopwords_removal=False, split=True):
    
    A = np.array(df['A'])
    A = [delete_non_alpha(text) for text in tqdm(A, desc='Deleting non-alpha characters')]
    A = [text.split() for text in A]

    if syntax_correction:
        sym_spell = setup_symspell()
        full_dict = get_full_dict()
        A = [correct_text(words,full_dict,sym_spell) for words in tqdm(A, desc='Correcting syntax')]

    if lemmatization:
        morph = pymorphy3.analyzer.MorphAnalyzer()
        A = [lemmatize(words, morph) for words in tqdm(A, desc='Lemmatizing')]
    
    if stopwords_removal:
        stopwords = get_stopwords()
        A = [delete_stop_words(words, stopwords) for words in tqdm(A, desc='Deleting stopwords')]
    
    A = [check_empty(words) for words in tqdm(A, desc='Checking empty text')]
    A = [text_from_words(words) for words in A]
    df['A'] = A
    df = df.dropna()
    if not split:
        return df
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Y'])
    return df_train, df_test