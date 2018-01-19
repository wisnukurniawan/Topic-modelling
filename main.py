import warnings
import re
from nltk.corpus import stopwords
import pandas as pd
import pyLDAvis.gensim
import gensim
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import logging
from gensim import corpora
from gensim.models import LdaModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud as wd

stopwords = set(stopwords.words('indonesia'))
stemmer = StemmerFactory().create_stemmer()

warnings.filterwarnings('ignore')


def cleaning(article):
    article = str(article)
    article = article.lower()
    article = re.sub(r"<(?!(?:a\s|/a|!))[^>]*>", " ", article)  # remove html tags and unicode characters
    article = removeEmoticons(article)  # remove emoticons
    article = re.sub(r'[^\x00-\x7f]', r' ', article)  # remove non-ascii
    article = re.sub("rt @\w+:", "", article)  # remove RT
    article = re.sub("rt@\w+:", "", article)  # remove RT
    article = re.sub(r"\b\d+\b", " ", article)  # remove digit or numbers
    article = re.sub(r"http\S+", "", article)
    article = re.sub(r"goo.gl\S+", "", article)
    article = re.sub(r"pic.\S+", "", article)
    article = ''.join([i for i in article if not i.isdigit()])
    article = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "",
                              article).split())  # menghilangkan semua tanda baca
    word_tokens = word_tokenize(article)
    article = " ".join([i for i in word_tokens if i not in stopwords])  # Menghilangkan stopwords
    article = " ".join(stemmer.stem(i) for i in article.split())  # stemming

    return article


def removeEmoticons(tweet):
    mycompile = lambda pat: re.compile(pat, re.UNICODE)
    # SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
    # MULTITOK_SMILEY = mycompile(r' : [\)dp]')
    NormalEyes = r'[:=]'
    Wink = r'[;]'
    NoseArea = r'(|o|O|-)'  ## rather tight precision, \S might be reasonable...
    HappyMouths = r'[D\)\]]'
    SadMouths = r'[\(\[]'
    Tongue = r'[pP]'
    OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

    Happy_RE = mycompile('(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
    Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

    Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
    Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
    Other_RE = mycompile('(' + NormalEyes + '|' + Wink + ')' + NoseArea + OtherMouths)

    Emoticon = (
        "(" + NormalEyes + "|" + Wink + ")" +
        NoseArea +
        "(" + Tongue + "|" + OtherMouths + "|" + SadMouths + "|" + HappyMouths + ")"
    )
    Emoticon_RE = mycompile(Emoticon)

    tweet = re.sub(Emoticon_RE, "", tweet)

    return tweet


df = pd.read_csv('documents/bolanet1.csv', sep=';')

print df['text']
print
text = df['text'].map(cleaning)
text_list = [i.split() for i in text]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename='running.log', filemode='w')

# Dictionary result
dictionary = corpora.Dictionary(text_list)
dictionary.save('dictionary.dict')
print "DICTIONARY"
print dictionary
print

doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word=dictionary, passes=50)
print doc_term_matrix

ldamodel.save('topic.model')
loading = LdaModel.load('topic.model')

d = gensim.corpora.Dictionary.load('dictionary.dict')
c = gensim.corpora.MmCorpus('corpus.mm')
lda = gensim.models.LdaModel.load('topic.model')

data = pyLDAvis.gensim.prepare(lda, c, d)
pyLDAvis.display(data)
pyLDAvis.save_html(data, 'topic_modeling.html')

for t in range(ldamodel.num_topics):
    plt.figure()
    plt.imshow(wd().fit_words(dict(ldamodel.show_topic(t, 200))))
    plt.axis("off")
    plt.title("Topic #" + str(t))

plt.show()
