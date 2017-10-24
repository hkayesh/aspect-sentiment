import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from core_classifier import CoreClassifier
from utilities import Utilities


class SentimentClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.core_classifier = CoreClassifier()
        self.sentiment_model = None
        self.utilities = Utilities()

    def train(self, training_file):
        input_data = self.utilities.get_segments_aspects_sentiments(training_file)
        X = self.utilities.convert_list_to_utf8(input_data['segments'])
        y = self.merge_classes(input_data['sentiments'])
        self.fit(X, y)

    def _make_prediction(self, model, segments):
        # segments = self.core_classifier.convert_list_to_utf8(segments)
        classes = model.labels_.classes_
        result = model.predict(segments)

        output_classes = [classes[class_id] for class_id in result]

        return output_classes

    def merge_classes(self, y):
        new_y = []
        for index, item in enumerate(y):
            if item != 'neutral':
                new_y.append(item)
            else:
                new_y.append('negative')

        return new_y

    def fit(self, X, y):
        X = self.utilities.convert_list_to_utf8(X)
        y = self.merge_classes(y)
        self.sentiment_model = self.core_classifier.get_classifier_model(X, y)

        return self

    def predict(self, X):
        X = self.utilities.convert_list_to_utf8(X)
        result = self._make_prediction(self.sentiment_model, X)

        return result


class AspectClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, casecade=False):
        self.casecade = casecade
        self.core_classifier = CoreClassifier()
        self.aspect_models = None
        self.utilities = Utilities()

    def train(self, training_file):
        input_data = self.utilities.get_segments_aspects_sentiments(training_file)

        X = input_data['segments']
        y = self.merge_classes(input_data['aspects'])
        self.fit(X, y)

    def _make_prediction_by_model(self, model, segments):
        classes = model.labels_.classes_
        result = model.predict(segments)

        output_classes = [classes[class_id] for class_id in result]

        return output_classes

    def _make_prediction(self, models, segments):
        result = []
        for segment in segments:
            aspect = 'other'
            for model in models:
                predicted_aspect = self._make_prediction_by_model(model, [segment])
                if predicted_aspect[0] != 'other':
                    aspect = predicted_aspect[0]
                    break
            result.append(aspect)

        return result

    def merge_classes(self, aspects):
        group_1 = ['staff attitude and professionalism', 'communication']
        group_2 = ['care quality', 'resource', 'process']
        group_3 = ['environment']
        group_4 = ['waiting time']
        group_5 = ['other', 'noise']
        group_6 = ['food']
        group_7 = ['parking']
        groups = [group_1, group_2, group_3, group_4, group_5, group_6, group_7]
        new_aspects = []
        for aspect in aspects:
            for group in groups:
                if aspect in group:
                    new_aspects.append(group[0])  # all members will be replaced by the first member of the group
                    break
        return new_aspects

    def get_classifier_model(self, classes, X, y):
        core_classifier = CoreClassifier()
        new_X = []
        new_y = []

        for index, item in enumerate(y):
            if item in classes:
                new_X.append(X[index])
                new_y.append(item)
            else:
                new_X.append(X[index])
                new_y.append('other')

        model = core_classifier.get_classifier_model(new_X, new_y)

        return model

    def fit(self, X, y):
        X = self.utilities.convert_list_to_utf8(X)
        y = self.merge_classes(y)
        waiting_time_model = self.get_classifier_model(['waiting time'], X, y)
        environment_model = self.get_classifier_model(['environment'], X, y)
        care_quality_model = self.get_classifier_model(['care quality'], X, y)
        staff_att_model = self.get_classifier_model(['staff attitude and professionalism'], X, y)
        multi_class_model = self.get_classifier_model(['waiting time', 'environment', 'care quality', 'staff attitude and professionalism'], X, y)
        # multi_class_model = self.get_classifier_model(set(y), X, y)

        if self.casecade == True:
            self.aspect_models = [staff_att_model, care_quality_model, waiting_time_model, environment_model]
        else:
            self.aspect_models = [multi_class_model]

        return self

    def predict(self, X):
        X = self.utilities.convert_list_to_utf8(X)
        result = self._make_prediction(self.aspect_models, X)

        return result
