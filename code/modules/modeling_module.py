import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF, TruncatedSVD
from spacy.en import STOP_WORDS as stopwords
import string
import spacy
nlp = spacy.load('en')
stopwords.update(['et', 'al', "'s", "—", '-', 'pp', 'pp.', 'p.'])
punctuations = string.punctuation


def spacy_tokenizer_1(doc):
    """
    A tokenizer called during vectorization

    INPUT:
    doc - the text to be tokenized

    OUTPUT:
    Tokenized text with the appropriate numbers removed
    """

    doc = doc.replace('—', '').replace(',', '').replace("'s", '')
    doc = re.sub('\s\d+\s', '', doc)
    tokens = nlp(doc)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else
              tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok
              not in punctuations)]
    return tokens


def spacy_tokenizer_2(doc):
    """
    A tokenizer called during vectorization

    INPUT:
    doc - the text to be tokenized

    OUTPUT:
    Tokenized text with the appropriate numbers removed
    """

    doc = doc.replace('—', '').replace(',', '').replace("'s", '')
    doc = re.sub('\s\d+(\s)?', '', doc)
    tokens = nlp(doc)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else
              tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok
              not in punctuations)]
    return tokens


def count_vectorizer(tokenizer, max_feat, X, ngram_start=1, ngram_stop=2,
                     max_df=0.6):
    """
    A function to apply CountVectorizer and the corresponding parameters

    INPUT:
    tokenizer - the spacy tokenizer to be applied
    max_feat - the number of maximum features to be used
    X - the data to be vectorized
    ngram_start - the start of ngram_range, set as a default to 1
    ngram_stop - the end of ngram_range, set as a default to 2
    max_df - max_df or the document frequency parameter

    OUTPUT:
    Vectorized matrix
    """

    vectorizer = CountVectorizer(tokenizer=tokenizer,
                                 ngram_range=(ngram_start, ngram_stop),
                                 max_df=max_df,
                                 max_features=max_feat)
    return vectorizer, vectorizer.fit_transform(X)


def tfidf_vectorizer(tokenizer, max_feat, X, ngram_start=1, ngram_stop=2,
                     max_df=0.6):
    """
    A function to apply TfidfVectorizer and the corresponding parameters

    INPUT:
    tokenizer - the spacy tokenizer to be applied
    max_feat - the number of maximum features to be used
    X - the data to be vectorized
    ngram_start - the start of ngram_range, set as a default to 1
    ngram_stop - the end of ngram_range, set as a default to 2
    max_df - max_df or the document frequency parameter

    OUTPUT:
    Vectorized matrix
    """

    vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                 ngram_range=(ngram_start, ngram_stop),
                                 max_df=max_df,
                                 max_features=max_feat)
    return vectorizer, vectorizer.fit_transform(X)


def lda_cv(X, n_comp, n_iter=10):
    """
    A function to apply LatentDirichletAllocation to a matrix

    INPUT:
    X - the data to be used for topic modeling
    n_comp - the number of desired topics
    n_iter - the number of iterations

    OUTPUT:
    The model used and the fit and transformed data
    """

    lda = LatentDirichletAllocation(n_components=n_comp,
                                    max_iter=n_iter,
                                    random_state=42,
                                    learning_method='online')
    return lda, lda.fit_transform(X)


def lda_tfidf(X, n_comp, n_iter=10):
    """
    A function to apply LatentDirichletAllocation to a matrix

    INPUT:
    X - the data to be used for topic modeling
    n_comp - the number of desired topics
    n_iter - the number of iterations

    OUTPUT:
    The model used and the fit and transformed data
    """

    lda = LatentDirichletAllocation(n_components=n_comp,
                                    max_iter=n_iter,
                                    random_state=42,
                                    learning_method='online')
    return lda, lda.fit_transform(X)


def lsa_tfidf(X, n_comp):
    """
    A function to apply TruncatedSVD to a matrix

    INPUT:
    X - the data to be used for topic modeling
    n_comp - the number of desired topics

    OUTPUT:
    The model used and the fit and transformed data
    """

    lsa = TruncatedSVD(n_components=n_comp, random_state=42)
    return lsa, lsa.fit_transform(X)


def lsa_cv(X, n_comp):
    """
    A function to apply TruncatedSVD to a matrix

    INPUT:
    X - the data to be used for topic modeling
    n_comp - the number of desired topics

    OUTPUT:
    The model used and the fit and transformed data
    """

    lsa = TruncatedSVD(n_components=n_comp, random_state=42)
    return lsa, lsa.fit_transform(X)


def nmf_tfidf(X, n_comp):
    """
    A function to apply NMF to a matrix

    INPUT:
    X - the data to be used for topic modeling
    n_comp - the number of desired topics

    OUTPUT:
    The model used and the fit and transformed data
    """

    nmf = NMF(n_components=n_comp, random_state=42)
    return nmf, nmf.fit_transform(X)


def nmf_cv(X, n_comp):
    """
    A function to apply NMF to a matrix

    INPUT:
    X - the data to be used for topic modeling
    n_comp - the number of desired topics

    OUTPUT:
    the model used and the fit and transformed data
    """

    nmf = NMF(n_components=n_comp, random_state=42)
    return nmf, nmf.fit_transform(X)


def display_topics(model, feature_names, no_top_words):
    """
    Displays word groupings from the latent space produced by topic modeling

    INPUT:
    model - the model used during topic modeling
    feature_names - feature_names() feature of the model
    no_top_words - the number of top words desired from each topic

    OUTPUT:
    A printed list of the topics
    """

    for ix, topic in enumerate(model.components_):
        print("Topic ", ix)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def use_vectorizer(cv_vect, cv_vect_trans, n_comp=10):
    """
    Uses a vectorizer and vectorized data to test three models print results

    INPUT:
    cv_vect - the model returned after vectorizing
    cv_vect_trans - the data after vectorizing
    n_comp - the number of 'topics' desired

    OUTPUT:
    A number of dimensions equal to n_comp
    """

    models = [lsa_cv, nmf_cv, lda_tfidf]
    names = ['LSA', 'NMF', 'LDA']
    i = 0
    for item in models:
        norm = Normalizer()
        vect_normalized = norm.fit_transform(cv_vect_trans)
        model, model_transformed = item(vect_normalized, n_comp=n_comp)
        print('\n\n-------' + names[i] + '------\n\n')
        display_topics(model, cv_vect.get_feature_names(), 10)
        i += 1
