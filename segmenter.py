from utilities import Utilities
from stanford import Stanford
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import tree
import numpy as np
import bisect
import pandas as pd
import math


class Segmenter(object):
    def __init__(self, training_file_path=None):
        self.utilities = Utilities()
        self.stanford = Stanford()
        self.missing_value = None
        self.encoders = []
        if training_file_path is not None:
            self.dataset_file_path = training_file_path
        else:
            # self.dataset_file_path = 'full_dataset.csv' # use it as a default dataset
            self.dataset_file_path = 'mmh_dataset.csv'  # use it only for aspect and sentiment evaluation
        self.comments = self.utilities.get_only_comments_from_dataset(self.dataset_file_path)
        self.features_and_labels = self.get_feature_matrix_from_reviews(self.comments)

        # self.model = RandomForestClassifier(n_estimators=100)
        # self.model = svm.SVR()
        self.model = tree.DecisionTreeClassifier()
        self._train()

    def get_sentences_with_multiple_segments(self, reviews):
        sentences_with_segments = []
        for review in reviews:
            sentences = self.utilities.split_text_into_insentence(review)
            for sentence in sentences:
                sentence = self.utilities.strip_nonalnum_re(re.sub("^\*\*\$\*\*|\*\*\$\*\*$", "", sentence))
                index = sentence.find("**$**")
                if 0 < index < len(sentence) and sentence not in sentences_with_segments:
                    sentences_with_segments.append(sentence)
        # self.utilities.write_content_to_file('segment_sentences.txt', "\n".join(sentences_with_segments))
        return sentences_with_segments

    def _get_candidate_positions(self, clauses):
        positions = []
        offset = 0
        del clauses[len(clauses) - 1]  # assuming there will be at least 2 clauses. Number of total positions is one less than the total clauses
        for clause in clauses:
            position = len(self.stanford.tokenize(clause)) + offset
            positions.append(position)
            offset = position
        return positions

    def _get_positive_positions(self, sentence):
        clauses = sentence.split('**$**')
        positions = self._get_candidate_positions(clauses)

        return positions

    def get_conjunction_positions(self, sentence):
        clauses_info = self.utilities.split_sentence_by_conjunction(sentence)

        indices = clauses_info['indices']

        positions = []
        for index in indices:
            substring = sentence[0:index]
            if substring:
                positions.append(len(self.stanford.tokenize(substring)))

        return positions

    def normalize_pos_tags(self, pos):
        adjectives = self.utilities.adjective_phrase_tags
        nouns = self.utilities.noun_phrase_tags
        adverbs = self.utilities.adverb_phrase_tags
        verbs = self.utilities.verb_phrase_tags

        if pos in adjectives:
            normalized_pos = 'JJ'
        elif pos in nouns:
            normalized_pos = 'NN'
        elif pos in adverbs:
            normalized_pos = 'RB'
        elif pos in verbs:
            normalized_pos = 'VB'
        else:
            normalized_pos = pos  # if not in the above category, return as it is

        return normalized_pos

    def _get_contaxt_featues(self, tokens, position, window_size, mode):
        context_features = []
        start = -int(window_size/2)
        end = -start
        for i in range(start, end+1):
            location = position + i
            if location < 0 or location > len(tokens) - 1:
                value = self.missing_value
            else:
                if mode == 'word':
                    value = self.stanford.get_stanford_lemma(tokens[location])
                elif mode == 'pos':
                    value = self.normalize_pos_tags(tokens[location])
                else:
                    value = tokens[location]

            context_features.append(value)

        return context_features

    def get_context_pos(self, sentence, tokens, position, window_size):
        tokens = self.stanford.tokenize(sentence)
        tagged_tokens = self.stanford.get_stanford_pos(sentence)
        all_pos = []
        for token in tokens:
            pos = 'NN' # default POS
            for token_pos_pair in tagged_tokens:
                if token == token_pos_pair:
                    pos = tagged_tokens[token]
                    break
            all_pos.append(pos)
        context_pos = self._get_contaxt_featues(all_pos, position, window_size, 'pos')

        return context_pos

    def get_closest_conjunction_distances(self, tokens, position):
        prev_conjunction_position = None
        next_conjunction_position = None

        for tok_position in range(0, position-1):
            if tokens[tok_position] in self.utilities.conjunctions:
                prev_conjunction_position = tok_position

        for tok_position in range(position + 1, len(tokens)):
            if tokens[tok_position] in self.utilities.conjunctions:
                next_conjunction_position = tok_position
                break
        positions = [prev_conjunction_position, next_conjunction_position]

        return positions

    def get_featuers_by_position(self, sentence, position):
        features = []
        window_size = 5
        tokens = self.stanford.tokenize(sentence)
        context_words = self._get_contaxt_featues(tokens, position, window_size, 'word')
        context_pos = self.get_context_pos(sentence, tokens, position, window_size)
        closest_conjunction_distances = self.get_closest_conjunction_distances(tokens, position)
        dependency_features = self.get_dependency_features(sentence, tokens, position)

        features = features + context_words + context_pos + closest_conjunction_distances + dependency_features

        return features

    def get_dependency_features(self, sentence, tokens, position):
        dependencies = self.stanford.get_stanford_dependencies(sentence)
        window_size = 5
        backward_dependencies = []
        start = position
        end = position - int(math.ceil(window_size/float(2)))
        for index in range(start, end, -1):
            for inner_index in range(index-1, end, -1):
                relation = None
                if 0 <= index < len(tokens) and 0 <= inner_index < len(tokens):
                    primary_word = tokens[index]
                    inner_word = tokens[inner_index]
                    if primary_word != inner_word:
                        deps_by_word = self.utilities.get_dependency_by_word(dependencies, tokens[index])  # dependencies by primary words of sentences
                        deps_by_inner_word = self.utilities.get_dependency_by_word(dependencies, tokens[inner_index])

                        if deps_by_word:
                            for dep in deps_by_word:
                                if dep in deps_by_inner_word:
                                    relation = dep[1]
                                    break
                backward_dependencies.append(relation)

        # forward_dependencies = []
        # end = position + int(math.ceil(window_size / float(2)))
        # for index in range(start, end):
        #     for inner_index in range(index+1, end):
        #         relation = None
        #         if index < len(tokens) and inner_index < len(tokens):
        #             primary_word = tokens[index]
        #             inner_word = tokens[inner_index]
        #             if primary_word != inner_word:
        #                 deps_by_word = self.utilities.get_dependency_by_word(dependencies, tokens[index])  # dependencies by primary words of sentences
        #                 deps_by_inner_word = self.utilities.get_dependency_by_word(dependencies, tokens[inner_index])
        #
        #                 if deps_by_word:
        #                     for dep in deps_by_word:
        #                         if dep in deps_by_inner_word:
        #                             relation = dep[1]
        #                             break
        #         forward_dependencies.append(relation)

        return backward_dependencies

    def extract_features(self, sentence):
        original_sentence = re.sub("\*\*\$\*\*", "", sentence)

        # token positions that have actual segment characters before them
        positive_positions = self._get_positive_positions(sentence)

        conjunction_positions = self.get_conjunction_positions(original_sentence)

        # token positions that does not have actual segment characters before them
        negative_positions = [item for item in conjunction_positions if item not in positive_positions]

        positive_data = []
        positive_label = []
        for positive_position in positive_positions:
            positive_data.append(self.get_featuers_by_position(original_sentence, positive_position))
            positive_label.append(1)  # label 1 = split and 0 = no split

        negative_data = []
        negative_label = []
        for negative_position in negative_positions:
            negative_data.append(self.get_featuers_by_position(original_sentence, negative_position))
            negative_label.append(0) # label 1 = split and 0 = no split

        items = {
            'data': positive_data + negative_data,
            'label': positive_label + negative_label
        }
        return items

    def transform_categorical_numerical(self, data, mode):
        numerical_data = []
        data = np.transpose(data)
        column_index = 0
        for column in data:
            if mode == 'train':
                encoder = LabelEncoder()
                numerical_data.append(encoder.fit_transform(column))
                self.encoders.append(encoder)
            elif mode == 'test':
                saved_encoder = self.encoders[column_index]
                column_index += 1

                # replaced new labels by '<unk>'
                column = pd.Series(column).map(
                    lambda s: '<unk>' if s not in saved_encoder.classes_ else s
                )

                saved_encoder_classes = list(saved_encoder.classes_)
                bisect.insort_left(saved_encoder_classes, '<unk>')
                saved_encoder.classes_ = saved_encoder_classes

                numerical_data.append(saved_encoder.transform(column))

        return np.transpose(numerical_data).tolist()

    def get_feature_matrix_from_reviews(self, reviews):
        sentences_with_multiple_segments = self.get_sentences_with_multiple_segments(reviews)
        data = []
        labels = []
        for sentence in sentences_with_multiple_segments:
            sentence = sentence.decode('utf-8')
            try:
                items = self.extract_features(str(sentence))
                data = data + items['data']
                labels = labels + items['label']
            except UnicodeEncodeError:
                # print "WARNING: skipped sentence for encoding issue - \"" + sentence + "\""
                continue
        data_matrix = {
            'data': data,
            'labels': labels
        }

        return data_matrix

    def _train(self):
        data = self.features_and_labels['data']
        labels = self.features_and_labels['labels']

        training_data_numerical = self.transform_categorical_numerical(data, 'train')
        self.model.fit(training_data_numerical, labels)

    def test(self, sentence, indices):
        results = []
        for index in indices:
            features = [self.get_featuers_by_position(sentence, index)]
            features = self.transform_categorical_numerical(features, 'test')
            result = self.model.predict(features)
            results.append(result[0])

        return results

    def get_segments(self, sentence):
        indices = self.get_conjunction_positions(sentence)
        result = self.test(sentence, indices)
        split_positions = []

        for i in range(0, len(result)):
            if result[i] == 1:
                split_positions.append(indices[i])

        tokens = self.stanford.tokenize(sentence)
        index = 0
        segmentes = {
            'positions': split_positions,
            'segments': []
        }

        for position in split_positions:
            segmentes['segments'].append(" ".join(tokens[index:position]))
            index = position+1

        if split_positions:
            segmentes['segments'].append(" ".join(tokens[index:len(tokens)]))
        else:
            segmentes['segments'] = [sentence]

        return segmentes
