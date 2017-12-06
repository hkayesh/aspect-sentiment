from utilities import Utilities
from comment_level_evaluation import CommentLevelEvaluation
import operator


class CombineSystems:
    def __init__(self):
        self.utilities = Utilities()

        self.storage_path = 'comment-level-datasets-2/'
        # self.storage_path = 'r-combine-outputs/'
        self.random_states = [111, 122, 133, 144, 155]
        self.categories = ['environment', 'waiting time', 'staff attitude professionalism', 'care quality', 'other']

    def is_valid_asp_from_from_system_a(self, aspect, confidence_value):
        is_valid = False
        thresholds = {'environment': 0.2,
                      'waiting time': 0.6,
                      'staff attitude and professionalism': 0.4,
                      'care quality': 0.1,
                      'other': 0.5,
                      }

        aspects = thresholds.keys()
        if aspect in aspects and float(confidence_value) >= thresholds[aspect]:
            is_valid = True

        return is_valid

    def is_valid_asp_from_from_system_b(self, aspect, confidence_value):
        is_valid = False
        thresholds = {'environment': 0.3,
                      'waiting time': 0.1,
                      'staff attitude and professionalism': 0.1,
                      'care quality': 0.1,
                      'other': 0.6
                      }

        aspects = thresholds.keys()
        if aspect in aspects and float(confidence_value) >= thresholds[aspect]:
            is_valid = True

        return is_valid

    def apply_dictionaries(self, comment):
        food_lexicon = ['food', 'canteen', 'canten', 'coffee', 'cofee', 'coffe', 'coffee', 'tea', 'drink', 'drinks']
        parking_lexicon = ['car park', 'car-park', 'carpark', 'parking', 'bicycle']

        aspects = []
        all_words = self.utilities.get_lemma(comment)
        lemmatized_words = all_words.values()

        for word in food_lexicon:
            if word in lemmatized_words:
                aspects.append('food')
                break

        for word in parking_lexicon:
            if word in lemmatized_words:
                aspects.append('parking')
                break

        return aspects

    def combine_by_dynamic_threshold(self, file_a_path, file_b_path, output_file_path):

        file_a = utilities.read_from_csv(file_a_path)
        file_b = utilities.read_from_csv(file_b_path)

        output = []
        for row_a, row_b in zip(file_a, file_b):

            comment = row_a[0]
            aspects = []

            # remove comment from the first column
            del row_a[0]
            del row_b[0]

            for a, b in zip(row_a, row_b):
                if not a and not b and a in self.categories:
                    break

                # union
                # if a and a not in aspects:
                #     aspects.append(a)
                #
                # if b and b not in aspects and b in self.categories:
                #     aspects.append(b)

                # intersection
                # if a and b and a == b and a not in aspects:
                #     aspects.append(a)

                # union with threshold
                if a is not None:
                    aspect_a = a.rsplit(' ', 1)[0]
                    if aspect_a not in aspects:
                        confidence_value_a = a.rsplit(' ', 1)[1]
                        is_valid = self.is_valid_asp_from_from_system_a(aspect_a, confidence_value_a)
                        if is_valid:
                            aspects.append(aspect_a)

                if b is not None:
                    aspect_b = b.rsplit(' ', 1)[0]
                    if aspect_b in self.categories and aspect_b not in aspects:
                        confidence_value_b = b.rsplit(' ', 1)[1]
                        is_valid = self.is_valid_asp_from_from_system_b(aspect_b, confidence_value_b)
                        if is_valid:
                            aspects.append(aspect_b)

            # Apply food and parking dictionaries
            # asps_from_dictionaries = self.apply_dictionaries(comment)
            # if len(asps_from_dictionaries) > 0:
            #     aspects = aspects + asps_from_dictionaries

            if len(aspects) < 1:
                aspects = ['other']

            output.append([comment] + aspects)
        self.utilities.save_list_as_csv(output, output_file_path)

    def combine_by_static_threshold(self, file_a_path, file_b_path, threshold_a, threshold_b, output_file_path):

        file_a = utilities.read_from_csv(file_a_path)
        file_b = utilities.read_from_csv(file_b_path)

        output = []
        for row_a, row_b in zip(file_a, file_b):

            comment = row_a[0]
            aspects = []

            # remove comment from the first column
            del row_a[0]
            del row_b[0]

            for a, b in zip(row_a, row_b):
                if not a and not b and a in self.categories:
                    break

                # union with threshold
                if a and a.rsplit(' ', 1)[0] not in aspects and float(a.rsplit(' ', 1)[1]) >= threshold_a:
                    aspects.append(a.rsplit(' ', 1)[0])

                if b and b.rsplit(' ', 1)[0] in self.categories and b.rsplit(' ', 1)[0] not in aspects and float(b.rsplit(' ', 1)[1]) >= threshold_b:
                    aspects.append(b.rsplit(' ', 1)[0])

            # Apply food and parking dictionaries
            # asps_from_dictionaries = self.apply_dictionaries(comment)
            # if len(asps_from_dictionaries) > 0:
            #     aspects = aspects + asps_from_dictionaries

            if len(aspects) < 1:
                aspects = ['other']

            output.append([comment] + aspects)

        self.utilities.save_list_as_csv(output, output_file_path)

    def extract_top_comments(self, data_file, output_file_path):
        rows = self.utilities.read_from_csv(data_file)

        envs = {}
        wts = {}
        saaps = {}
        cqs = {}
        ots = {}

        for row in rows:
            comment = row[0]
            del rows[0]

            for item in row:
                if item and item.rsplit(' ', 1)[0] == 'environment':
                    envs[comment] = float(item.rsplit(' ', 1)[1])

                if item and item.rsplit(' ', 1)[0] == 'waiting time':
                    wts[comment] = float(item.rsplit(' ', 1)[1])

                if item and item.rsplit(' ', 1)[0] == 'staff attitude and professionalism':
                    saaps[comment] = float(item.rsplit(' ', 1)[1])

                if item and item.rsplit(' ', 1)[0] == 'care quality':
                    cqs[comment] = float(item.rsplit(' ', 1)[1])

                if item and item.rsplit(' ', 1)[0] == 'other':
                    ots[comment] = float(item.rsplit(' ', 1)[1])

        # sort comments by the descending order of confidence values
        sorted_envs = [comment_data[0] for comment_data in sorted(envs.items(), key=operator.itemgetter(1), reverse=True)]
        sorted_wts = [comment_data[0] for comment_data in sorted(wts.items(), key=operator.itemgetter(1), reverse=True)]
        sorted_saaps = [comment_data[0] for comment_data in sorted(saaps.items(), key=operator.itemgetter(1), reverse=True)]
        sorted_cqs = [comment_data[0] for comment_data in sorted(cqs.items(), key=operator.itemgetter(1), reverse=True)]
        sorted_ots = [comment_data[0] for comment_data in sorted(ots.items(), key=operator.itemgetter(1), reverse=True)]

        # prepare output to save
        output = [['Environment', 'Waiting time', 'Staff attitude and professionalism', 'Care quality', 'Other']]
        top = 5
        for i in range(0, top):
            comments = []

            try:
                comments.append(sorted_envs[i])
            except IndexError:
                comments.append(None)

            try:
                comments.append(sorted_wts[i])
            except IndexError:
                comments.append(None)

            try:
                comments.append(sorted_saaps[i])
            except IndexError:
                comments.append(None)

            try:
                comments.append(sorted_cqs[i])
            except IndexError:
                comments.append(None)

            try:
                comments.append(sorted_ots[i])
            except IndexError:
                comments.append(None)

            output.append(comments)
        self.utilities.save_list_as_csv(output, output_file_path)


combine_systems = CombineSystems()
com_eval = CommentLevelEvaluation()
utilities = Utilities()

# calculate per system scores
# com_eval.calculate_per_system_accuracy('mmhsct')

# calculate combine evaluation scores to find best threshold combination
# thresholds_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# for a in thresholds_a:
#     thresholds_b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
#     for b in thresholds_b:
#         random_states = [111, 122, 133, 144, 155]
#         for random_state in random_states:
#             file_a_path = 'comment-level-datasets-2/mmhsct_output_' + str(random_state) + '.csv'
#             file_b_path = 'r-combine-outputs/mmhsct_output_confidence_' + str(random_state) + '.csv'
#             output_file_path = 'r-combine-outputs/mmhsct_combined_confidence_' + str(random_state) + '.csv'
#
#             combine_systems.combine_by_static_threshold(file_a_path, file_b_path, a, b, output_file_path)
#
#         # test_file_path = 'comment-level-datasets-2/mmhsct_test_111.csv'
#         com_eval.calculate_accuracy('mmhsct')
#         # com_eval.calculate_per_system_accuracy('mmhsct')
#         # combine_systems.extract_top_comments(file_b)

# apply best thresholds for combination
random_states = [111, 122, 133, 144, 155]
for random_state in random_states:
    file_a_path = 'comment-level-datasets-2/mmhsct_output_' + str(random_state) + '.csv'
    file_b_path = 'r-combine-outputs/mmhsct_output_confidence_' + str(random_state) + '.csv'
    output_file_path = 'r-combine-outputs/mmhsct_combined_confidence_' + str(random_state) + '.csv'

    combine_systems.combine_by_dynamic_threshold(file_a_path, file_b_path, output_file_path)
com_eval.calculate_accuracy('mmhsct')

