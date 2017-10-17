from segmenter import Segmenter
from sklearn import tree
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report as clsr
from sklearn.model_selection import train_test_split as tts, StratifiedKFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import itertools
from sklearn import metrics
from utilities import Utilities
from processing import Processor
from stanford import Stanford

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class Evaluator:
    def __init__(self, data_file):
        self.utilities = Utilities()
        self.data_file = data_file
        self.processor = Processor({'training_file': data_file})
        self.segmenter = self.processor.load_segmenter()
        self.stanford = Stanford()
        self.segments = []
        self.aspects = []
        self.sentiments = []
        self.prepare_aspect_sentiment_data()

    def calculate_evaluatio_matrices(self, labels, result):
        positives = 0
        negatives = 0

        for label in labels:
            if label == 1:
                positives += 1
            elif label == 0:
                negatives += 1

        evaluation_info = {
            'positives': positives,
            'negatives': negatives,
            # 'precision': "%.3f" % precision_score(labels, result),
            # 'recall': "%.3f" % recall_score(labels, result),
            'accuracy': "%.3f" % accuracy_score(labels, result),
            'f1_score': "%.3f" % recall_score(labels, result)
        }

        return evaluation_info

    def evaluate_segmentation(self):
        dataset = self.segmenter.features_and_labels
        all_data_transformed = self.segmenter.transform_categorical_numerical(dataset['data'], 'train')
        all_data_unique = self.utilities.get_unique_list_of_lists(all_data_transformed, dataset['labels'])

        # model = SGDClassifier()
        model = svm.SVC(kernel='linear')
        # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1) # Neural Network
        # model = MultinomialNB()
        # model = RandomForestClassifier(random_state=5)
        # model = tree.DecisionTreeClassifier(random_state=0)

        X = all_data_unique['data']
        y = all_data_unique['labels']

        f1_scores = cross_val_score(model, X, y, scoring='f1_micro', cv=5)
        print [round(score, 3) for score in f1_scores.tolist()]
        print("F1-score: %0.4f" % (f1_scores.mean()))

    def prepare_aspect_sentiment_data(self):
        data = self.utilities.get_segments_aspects_sentiments(self.data_file)
        self.segments = data['segments']
        self.aspects = data['aspects']
        self.sentiments = data['sentiments']

    def transform_aspect_name_list(self, name_list):
        id_list = []

        for name in name_list:
            id_list.append(self.lexicon.get_aspect_id_by_name(name))

        return id_list

    def evaluate_classifier(self, classifier, X, y):

        # Begin evaluation
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=5)
        model = classifier.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # *** save info for error analysis
        errors = []
        for index in range(0, len(X_test)):
            if y_test[index] != y_pred[index]:
                errors.append("\""+X_test[index] +"\",\""+ y_test[index]  +"\",\""+ y_pred[index]+"\"")

        str_out = "\n".join(errors)
        self.utilities.write_content_to_file('aspect_errors.csv', str_out)


        print(clsr(y_test, y_pred))


    def evaluate_aspect_extraction(self):
        X = self.segments
        y = self.aspects

        self.evaluate_classifier(self.processor.ml_asp_classifier, X, y)

    def transform_sentiment_classes(self, sentiment_names):
        sentiment_values = []
        for sentiment_name in sentiment_names:
            sentiment_values.append(self.utilities.sentiment_classes.index(sentiment_name))

        return sentiment_values

    def evaluate_sentiment_detection(self):
        X = self.segments
        y = self.sentiments

        # y = self.processor.ml_snt_classifier.merge_classes(y)
        self.evaluate_classifier(self.processor.ml_snt_classifier,X, y)



