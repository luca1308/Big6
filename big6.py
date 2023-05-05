# %% [markdown]
# # Transforming the VOSViewer data

# %%
import json
import codecs
import pandas as pd

filename = 'VOSviewer.json'

data = json.load(codecs.open(filename, 'r', 'utf-8-sig'))
items = data["network"]["items"]
vos = pd.DataFrame(items)[['label', 'description', 'url', 'cluster']]
vos.sort_values(by=['cluster'])

vos["assigned to"] = ''
names = ["Alessandro", "Albana", "Edoardo", "Riccardo", "Simone", "Luca"]
l = int(len(vos)/len(names)) + (len(vos) % len(names) > 0)
n = 0
i = 0
c = 1

for i in range(0, len(vos)):
    vos["assigned to"][i] = names[n]
    c += 1
    if c > l:
        n += 1
        c = 1

vos.to_excel("list_of_papers.xlsx")
vos


# %% [markdown]
# # Conducting NLP

# %%
import pandas as pd
import re

filename2 = "abstracts.xlsx"
abstracts_df = pd.read_excel(filename2, index_col=0)
abstracts_df.sort_values(by=["cluster"], inplace=True)
abstracts_df


# %%
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = stopwords.words("english")


# %%
pos_result_test = []
for text in abstracts_df.abstract:
    h = []
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    
    # Remove words with unimportant Part of Speech
    [h.append(word) for (word, pos) in tag if pos != 'VB' and pos != 'VBD' and pos != 'VBG' and pos != 'VBN' and
     pos != 'VBP' and pos != 'RB' and pos != 'RBR' and pos != 'RBS' and pos != 'WRB' and pos != 'DT' and pos != 'TO'
     and pos != 'DT' and pos != 'VBZ' and pos != 'CC' and pos != 'IN']
    pos_result_test.append(h)


def listToString(s):
    str1 = " "
    return (str1.join(s))

# format transformation
corpus_pos = []
for i in pos_result_test:
    text = listToString(i)
    corpus_pos.append(text)


# %%
from sklearn.feature_extraction.text import CountVectorizer

candidates_list = []
n_gram_range = (2, 3)
for i in range(len(corpus_pos)):
    # Noise
    doc = re.sub(r"[0-9]+", "", corpus_pos[i]).lower()
    doc = re.sub('[^\w\s]', " ", doc)
    doc = re.sub('\s+', ' ', doc)
    doc = [doc]

    # Extract words/phrases, remove stopwords
    count = CountVectorizer(ngram_range=n_gram_range,
                            stop_words=stop_words).fit(doc)
    candidates = count.get_feature_names_out()

    # Remove candidates not in doc
    # Remove candidates which end with adjectives
    candidates_out = []
    for j in range(len(candidates)):
        if abstracts_df.abstract[i].lower().find(candidates[j]) > 0:
            tokens = nltk.word_tokenize(candidates[j])
            tag = nltk.pos_tag(tokens)
            if tag[-1][1] != 'JJ' and tag[-1][1] != 'JJR' and tag[-1][1] != 'JJS':
                candidates_out.append(candidates[j])
            else:
                print(i, "Remove adj:", candidates[j])
        else:
            print(i, "Remove:", candidates[j])
    candidates_list.append(candidates_out)


# %%
# selected key phrase amount from each abstract
for i in range(len(candidates_list)):
    print(len(candidates_list[i]))
    
# total amount of candidate key phrases
length = 0
for i in range(len(candidates_list)):
    length = (length+len(candidates_list[i]))
print(length)


# %%
candidates



