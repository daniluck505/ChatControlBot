import dill
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


class Model:
    def __init__(self):
        self.model = None
        self.filename = "model.pkl"

    def load(self):
        with open(self.filename, 'rb') as f:
            self.model = dill.load(f)
        print(self.model)

    def dump(self):
        with open(self.filename, 'wb') as f:
            dill.dump(self.model, f)
        print('dump finish')

    def make_model(self):
        print('download dataset.txt')
        data_list = []
        with open("dataset.txt", 'r') as file:
            for line in file:
                # print(line)
                labels = line.split()[0]
                text = line[len(labels) + 1:].strip()
                labels = labels.split(",")
                mask = [1 if "__label__NORMAL" in labels else 0,
                        1 if "__label__INSULT" in labels else 0,
                        1 if "__label__THREAT" in labels else 0,
                        1 if "__label__OBSCENITY" in labels else 0]
                data_list.append((text, *mask))
        df = pd.DataFrame(data_list, columns=["text", "normal", "insult", "threat", "obscenity"])

        print('decode_label')

        def decode_label(x):
            if x['normal']:
                return 1
            else:
                return 0
        df['label'] = df[["normal", "insult", "threat", "obscenity"]].apply(decode_label, axis=1)
        print('make new_df')
        new_df = df.drop(columns=["normal", "insult", "threat", "obscenity"], axis=1)
        snowball = SnowballStemmer(language='russian')
        stop_words_russian = stopwords.words('russian')

        def tokenize_text(text, del_stop_words=True):
            tokens = word_tokenize(text, language='russian')
            tokens = [i for i in tokens if i not in string.punctuation]
            if del_stop_words:
                tokens = [i for i in tokens if i not in stop_words_russian]
            tokens = [snowball.stem(i) for i in tokens]
            return tokens
        print('make Pipeline')
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_text(x, del_stop_words=True))),
            ('model', LogisticRegression())
        ])
        print('model fit')
        self.model.fit(new_df['text'], new_df['label'])
        print('finish')

    def pred(self, text):
        return self.model.predict([text])[0]

    def pred_prob_agress(self, text):
        return round(self.model.predict_proba([text])[0][0], 3)
