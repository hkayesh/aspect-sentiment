import os
import string
import pickle
import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter
from utilities import Utilities

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


class NLTKPreprocessor():
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


class CoreClassifier(object):

    def __init__(self):
        self.labels = LabelEncoder()
        # self.classifier = SGDClassifier()
        self.classifier = svm.SVC(kernel='linear', probability=True)
        # self.classifier = MultinomialNB()
        # self.classifier = RandomForestClassifier()
        # self.classifier = tree.DecisionTreeClassifier()

    def _build(self, classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        from sklearn.multiclass import OneVsRestClassifier
        classifier = OneVsRestClassifier(classifier)
        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 2))),
            ('classifier', classifier),
        ])
        model.fit(X, y)

        return model

    def get_classifier_model(self, X, y, outpath=None):
        """
        Builds a classifer for the given list of documents and targets in two
        stages: the first does a train/test split and prints a classifier report,
        the second rebuilds the model on the entire corpus and returns it for
        operationalization.
    
        X: a list or iterable of raw strings, each representing a document.
        y: a list or iterable of labels, which will be label encoded.
    
        Can specify the classifier to build with: if a class is specified then
        this will build the model with the Scikit-Learn defaults, if an instance
        is given, then it will be used directly in the build pipeline.
    
        If outpath is given, this function will write the model as a pickle.
        If verbose, this function will print out information to the command line.
        """
        # Label encode the targets
        y = self.labels.fit_transform(y)
        model= self._build(self.classifier, X, y)
        model.labels_ = self.labels

        if outpath is not None:
            with open(outpath, 'wb') as f:
                pickle.dump(model, f)

            print("Model written out to {}".format(outpath))

        return model

    def load_saved_model(self, model_path):
        model = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        return model

    def show_most_informative_features(self, model, text=None, n=20):
        """
        Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
        the n most informative features of the model. If text is given, then will
        compute the most informative features for classifying that text.
    
        Note that this function will only work on linear models with coefs_
        """
        # Extract the vectorizer and the classifier from the pipeline
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']

        # Check to make sure that we can perform this computation
        if not hasattr(classifier, 'coef_'):
            raise TypeError(
                "Cannot compute most informative features on {} model.".format(
                    classifier.__class__.__name__
                )
            )

        if text is not None:
            # Compute the coefficients for the text
            tvec = model.transform([text]).toarray()
        else:
            # Otherwise simply use the coefficients
            tvec = classifier.coef_

        # Zip the feature names with the coefs and sort
        coefs = sorted(
            zip(tvec[0], vectorizer.get_feature_names()),
            key=itemgetter(0), reverse=True
        )

        topn  = zip(coefs[:n], coefs[:-(n+1):-1])

        # Create the output string to return
        output = []

        # If text, add the predicted value to the output.
        if text is not None:
            output.append("\"{}\"".format(text))
            output.append("Classified as: {}".format(model.predict([text])))
            output.append("")

        # Create two columns with most negative and most positive features.
        for (cp, fnp), (cn, fnn) in topn:
            output.append(
                "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
            )

        return "\n".join(output)

    def get_cascade_models(self):
        model_paths = ['waiting_time_model.pickle',
                       'environment_model.pickle',
                       'staff_att_model.pickle',
                       'care_quality_model.pickle']

        models = []
        for model_path in model_paths:
            model = self.load_saved_model(model_path)
            if model is not None:
                models.append(model)
        return models
