from utilities import Utilities
from sklearn.model_selection import train_test_split as tts, StratifiedKFold

from processing import Processor


class CommentLevelEvaluation:

    def __init__(self, data_file):

        self.data_file = data_file
        self.utilities = Utilities()
        # self.Processor = Processor()

        self.storage_path = 'evaluation_datasets/'
        self.random_states = [11, 22, 33, 44, 55]

    def generate_datasets(self):
        X = self.utilities.read_from_csv(self.data_file)
        y = [0] * len(X)  # fake labels
        for random_state in self.random_states:
            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=random_state)

            for row in X_test:
                row[0] = row[0].replace('**$**', "")

            self.utilities.save_list_as_csv(X_train, self.storage_path + 'mmhsct_train_' + str(random_state) +'.csv')
            self.utilities.save_list_as_csv(X_test, self.storage_path + 'mmhsct_test_' + str(random_state) +'.csv')

    def run_experiment(self):
        for random_state in self.random_states:
            X_train = 'mmhsct_train_' + str(random_state) + '.csv'
            X_test = 'mmhsct_test_' + str(random_state) + '.csv'

            settings = {
                'training_file': X_train,
                'data_file': X_test,
                'max_reviews': None,  # Options: 0 to any integer | default: None (all)
                'output_file': 'srft_output_' + str(random_state) + '.csv'
            }

            processor = Processor(settings=settings)
            processor.run()
            exit()

    def merge_aspect_classes(self, aspects):
        group_1 = ['staff attitude and professionalism', 'communication']
        group_2 = ['care quality', 'resource', 'process']
        group_3 = ['environment', 'food', 'parking']
        group_4 = ['waiting time']
        group_5 = ['other', 'noise']
        groups = [group_1, group_2, group_3, group_4, group_5]
        new_aspects = []
        for aspect in aspects:
            for group in groups:
                if aspect in group:
                    new_aspects.append(group[0])  # all members will be replaced by the first member of the group
                    break
        return new_aspects

    def calculate_comment_level_scores_for_categories(self, y_test, y_pred):
        categories = []
        for aspects in y_test:
            categories = categories + aspects
        categories = list(set(categories))
        category_f_scores = {}
        for category in categories:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0

            for index, test_categories in enumerate(y_test):
                pred_categories = y_pred[index]

                if category in test_categories and category in pred_categories:
                    true_positives += 1
                elif category in test_categories and category not in pred_categories:
                    false_negatives += 1
                elif category not in test_categories and category in pred_categories:
                    false_positives += 1
                else:
                    true_negatives += 1

            # print [true_positives, false_positives, false_negatives, true_negatives]
            precision = true_positives / float(true_positives + false_positives)
            recall = true_positives / float(true_positives + false_negatives)

            f_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            category_f_scores[category] = f_score

        return category_f_scores





    def calculate_accuracy(self):
        for random_state in self.random_states:
            # X_test = self.utilities.read_from_csv('evaluation_datasets/mmhsct_test_' + str(random_state) + '.csv')
            # X_pred = self.utilities.read_from_csv('evaluation_datasets/mmhsct_output_' + str(random_state) + '.csv')
            X_test = self.utilities.read_from_csv('evaluation_datasets/srft_test_' + str(random_state) + '.csv')
            X_pred = self.utilities.read_from_csv('evaluation_datasets/srft_output_' + str(random_state) + '.csv')

            y_test = []
            y_pred = []
            for index, row in enumerate(X_test):
                del row[0]
                aspects = []
                for item in row:
                    if item:
                        aspects.append(item.rsplit(' ', 1)[0])
                y_test.append(list(set(self.merge_aspect_classes(aspects))))

                predicted_row = X_pred[index]

                del predicted_row[0]
                aspects = []
                for item in predicted_row:
                    if item:
                        aspects.append(item.rsplit(' ', 1)[0])
                y_pred.append(list(set(aspects)))

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0

            for index, test in enumerate(y_test):
                pred = y_pred[index]

                pred_minus_test = [item for item in pred if item not in test]
                test_minus_pred = [item for item in test if item not in pred]

                if len(pred_minus_test) == 0 and len(test_minus_pred) == 0:
                    true_positives += 1
                # elif len(pred_minus_test) > 0 and len(test_minus_pred) == 0:
                elif len(pred_minus_test) > 0:
                    false_positives += 1
                # elif len(test_minus_pred) > 0 and len(pred_minus_test) == 0:
                elif len(test_minus_pred) > 0:
                    false_negatives += 1
                else:
                    true_negatives += 1

            precision = true_positives / float(true_positives + false_positives)
            recall = true_positives / float(true_positives + false_negatives)

            overall_f_score = (2*precision * recall) / (precision + recall)
            overall_accuracy = (true_positives + true_negatives) / float(len(y_test))

            print overall_accuracy
            # print self.calculate_comment_level_scores_for_categories(y_test, y_pred)

evaluation = CommentLevelEvaluation('mmh_dataset.csv')
# evaluation = CommentLevelEvaluation('sr_dataset.csv')

# evaluation.generate_datasets()
# evaluation.run_experiment(settings)

# from processing import Processor
# random_state = 11
# settings = {
#     'training_file': 'srft_train_'+str(random_state)+'.csv',
#     'data_file': 'srft_test_'+str(random_state)+'.csv',
#     'max_reviews': None,  # Options: 0 to any integer | default: None (all)
#     'output_file': 'srft_output_'+str(random_state)+'.csv'
# }
#
# processor = Processor(settings=settings)
# processor.run()

evaluation.calculate_accuracy()