# %% [markdown]
# # Big6 Data Science part of project
# You can run this project using your jupyter IDE by creating an environment in anaconda based on the provided yaml-file called "big6.yml".<br>
# Just execute <br>
# `conda env create -f [PATH_TO_FILE_ON_YOUR_MACHINE]/big6.yml`<br>
# within your anaconda terminal

# %% [markdown]
# ##  1. Transforming the VOSviewer data

# %%
import json
import codecs
import pandas as pd

filename = "VOSviewer/VOSviewer.json"

data = json.load(codecs.open(filename, "r", "utf-8-sig"))
items = data["network"]["items"]
vos = pd.DataFrame(items)[["label", "description", "url", "cluster"]]
vos.sort_values(by=["cluster"])

vos["assigned to"] = ""
names = ["Alessandro", "Albana", "Edoardo", "Riccardo", "Simone", "Luca"]
l = int(len(vos) / len(names)) + (len(vos) % len(names) > 0)
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
# ## 2. Conducting NLP to find important bi-grams and rank relevance of papers within universe by number of bi-grams within abstracts

# %%
import re

filename2 = "VOSviewer/abstracts.xlsx"
abstracts_df = pd.read_excel(filename2, index_col=0)
abstracts_df.sort_values(by=["cluster"], inplace=True)
abstracts_df


# %%
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")
stop_words = stopwords.words("english")
stop_words.extend(["abstract", "paper"])


# %%
pos_test = []
for text in abstracts_df.abstract:
    h = []
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)

    # Unimportant
    [
        h.append(word)
        for (word, pos) in tag
        if pos != "VB"
        and pos != "VBD"
        and pos != "VBG"
        and pos != "VBN"
        and pos != "VBP"
        and pos != "RB"
        and pos != "RBR"
        and pos != "RBS"
        and pos != "WRB"
        and pos != "DT"
        and pos != "TO"
        and pos != "DT"
        and pos != "VBZ"
        and pos != "CC"
        and pos != "IN"
    ]
    pos_test.append(h)


def listToString(s):
    str1 = " "
    return str1.join(s)


corpus_list = []
for i in pos_test:
    text = listToString(i)
    corpus_list.append(text)

# %%
from sklearn.feature_extraction.text import CountVectorizer

keyphrase_list = []
n_gram_range = (2, 3)
for i in range(len(corpus_list)):
    doc = re.sub(r"[0-9]+", "", corpus_list[i]).lower()
    doc = re.sub("[^\w\s]", " ", doc)
    doc = re.sub("\s+", " ", doc)
    doc = [doc]

    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(doc)
    keyphrase = count.get_feature_names_out()

    keyphrase_out = []
    for j in range(len(keyphrase)):
        if abstracts_df.abstract[i].lower().find(keyphrase[j]) > 0:
            tokens = nltk.word_tokenize(keyphrase[j])
            tag = nltk.pos_tag(tokens)
            if tag[-1][1] != "JJ" and tag[-1][1] != "JJR" and tag[-1][1] != "JJS":
                keyphrase_out.append(keyphrase[j])
            else:
                print(i, "Remove adj:", keyphrase[j])
        else:
            print(i, "Remove:", keyphrase[j])
    keyphrase_list.append(keyphrase_out)

# %%
length = 0
for i in range(len(keyphrase_list)):
    length = length + len(keyphrase_list[i])

print("List of keyphrase: \n", keyphrase)

# %%
remaining_keywords_lists = [x for x in keyphrase_list if x]
remaining_keywords_list = [
    item for sublist in remaining_keywords_lists for item in sublist
]
remaining_keywords_list

# %%
abstracts_df["keywords"] = keyphrase_list
keywords = abstracts_df[abstracts_df["keywords"].map(lambda d: len(d)) > 0]
s = keywords.keywords.str.len().sort_values(ascending=False).index
keyword_df = abstracts_df.reindex(s)
keyword_df.reset_index(drop=True, inplace=True)
keyword_df.to_excel("by_keyword.xlsx")
keyword_df


# %% [markdown]
# ## 3. Visualizations

# %%
import matplotlib.pyplot as plt
%matplotlib inline


# %% [markdown]
# ### 3.1 Wordcloud

# %%
from wordcloud import WordCloud, STOPWORDS

def cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off")


wordcloud = WordCloud(
    width=2000,
    height=1000,
    random_state=42,
    background_color="white",
    colormap="plasma",
    collocations=False,
    stopwords=STOPWORDS,
).generate(" ".join(corpus_list))

cloud(wordcloud)

# %% [markdown]
# ### 3.2 Countplot of relevant keyphrases

# %%
import seaborn as sns

# Most relevant bi-grams
unique_df = pd.DataFrame(data={"keyword": remaining_keywords_list})
sns.set(style="darkgrid")
plt.figure(figsize=(20, 15))
ax = sns.countplot(
    data=unique_df,
    x="keyword",
    order=pd.value_counts(unique_df.keyword).iloc[:16].index,
)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)

# %% [markdown]
# ## 4. Topic Modeling with Gensim

# %%
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from spacy.cli.download import download
import pyLDAvis
import pyLDAvis.gensim_models

# %%
def sent2words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


data_words = list(sent2words(corpus_list))

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
print(trigram_mod[bigram_mod[data_words[0]]])

# %%
def stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out

# %%
nostops = stopwords(data_words)
bigrams = bigrams(nostops)

nlp = spacy.load("en_core_web_sm")

data_lemmatized = lemmatization(bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"])

id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

# %%
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=5,
    random_state=42,
    update_every=1,
    chunksize=50,
    passes=500,
    alpha="auto",
    per_word_topics=True,
)

# %%
print("\nPerplexity: ", lda_model.log_perplexity(corpus))

coherence_lda = CoherenceModel(
    model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence="c_v"
)
coherence_lda = coherence_lda.get_coherence()

print("\nCoherence: ", coherence_lda)


# %%
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
vis



