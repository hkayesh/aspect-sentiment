from utilities import Utilities
from wrapper_classifiers import AspectClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.metrics import f1_score, accuracy_score


class SegmentLevelEvaluation:
    def __init__(self):
        self.utilities = Utilities()
        self.aspect_classifier = AspectClassifier()
        self.random_states = [11, 22, 33, 44, 55]


    def run_experiment(self):
        path = 'segment-level-datasets/'
        database = 'mmhsct'
        # database = 'srft'

        for random_state in self.random_states:
            training_file = path + database + '_segments_train_' + str(random_state) + '.csv'
            test_file = path + database + '_segments_test_' + str(random_state) + '.csv'

            training_data = self.utilities.read_from_csv(training_file)
            X_train = []
            y_train = []

            for row in training_data:
                X_train.append(row[0])
                y_train.append(row[1])

            test_data = self.utilities.read_from_csv(test_file)
            X_test = []
            y_test = []

            for row in test_data:
                X_test.append(row[0])
                y_test.append(row[1])

            model = self.aspect_classifier.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print clsr(y_test, y_pred, digits=4)
            print accuracy_score(y_test, y_pred)

seg_eval = SegmentLevelEvaluation()
seg_eval.run_experiment()
