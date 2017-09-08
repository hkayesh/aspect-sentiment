import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import csv
import pickle
import re


class Utilities(object):

    def __init__(self):
        self.noun_phrase_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        self.verb_phrase_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.adjective_phrase_tags = ['JJ', 'JJS', 'JJR']
        self.adverb_phrase_tags = ['RB', 'RBR', 'RBS']
        self.sentiment_classes = ['negative', 'neutral', 'positive']
        self.conjunctions_file_path = 'conjunctions.txt'

        self.conjunctions = self.get_lines_from_text_file(self.conjunctions_file_path)

    def write_content_to_file(self, file_path, content):
        output_file = open(file_path, 'w')
        output_file.write(content)
        output_file.close()

    def get_lines_from_text_file(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        content = [line.strip() for line in lines]

        return content

    def split_text_into_insentence(self, text):
        sentences = sent_tokenize(text)

        return sentences

    def read_from_csv(self, file_path):
        data = []

        with open(file_path, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data.append(row)

        return data

    def store_list_to_file(self,file_path, data_list):
        with open(file_path, 'wb') as f:
            pickle.dump(data_list, f)

    def get_list_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)

        return data_list

    def split_sentence_by_conjunction(self, sentence):
        sentence = sentence.lower()
        clauses = [sentence]
        matched_conjunctions = []
        all_indices = []
        for conjunction in self.conjunctions:
            if len(sentence.split(conjunction)) > 1:
                matched_conjunctions.append(conjunction)
                # TODO: fix the regex for to back to back to conjunctions
                iter = re.finditer(r"(?:^|\W)"+conjunction.lower()+"(?:$|\W)", sentence)
                indices = [m.start(0) for m in iter]
                # print indices
                all_indices = all_indices + indices

        all_indices = sorted(list(set(all_indices)))
        # temp_sentence = sentence
        for matched_conjunction in matched_conjunctions:
            # match with conjunction (whole words only)
            substrs = re.compile(r"(?:^|\W)"+matched_conjunction.lower()+"(?:^|\W)").split(sentence)
            sentence = '**$**'.join(substrs)

            clauses = filter(None, sentence.split('**$**'))
            index = 0
            for clause in clauses:
                clauses[index] = clause.strip()
                index += 1
        clause_info = {
            'clause': clauses,
            'indices': all_indices
        }
        return clause_info

    def get_only_comments_from_dataset(self, dataset_file_path):
        rows = self.read_from_csv(dataset_file_path)
        del rows[0]
        comments = []
        for row in rows:
            if len(row) > 2 and row[0] not in comments: # making comments list unique
                comments.append(row[0])

        return comments

    def get_unique_list_of_lists(self, data, labels):
        new_data = []
        new_labels = []
        index = 0
        for elem in data:
            if elem not in new_data:
                new_data.append(elem)
                new_labels.append(labels[index])
            index += 1

        unique_data = {
            'data': new_data,
            'labels': new_labels
        }

        return unique_data

    def get_dependency_by_relation(self, dependencies, relation):
        dependency_list = []
        for dependency in dependencies:
            if dependency[1] == relation:
                dependency_list.append(dependency)
        return dependency_list

    def get_dependency_by_word(self, dependencies, word):
        dependency_list = []
        for dependency in dependencies:
            if dependency[0][0] == word or dependency[2][0] == word:
                dependency_list.append(dependency)

        return dependency_list

    def strip_nonalnum_re(self, word):
        return re.sub(r"^\W+|\W+$", "", word)

    def get_segments_aspects_sentiments(self, dataset_file_path):
        segments = []
        aspects = []
        sentiments = []

        reviews = self.read_from_csv(dataset_file_path)
        segment_aspect_pairs = []

        for review in reviews:
            comment = review[0]
            comment_parts = comment.split('**$**')
            index = 1
            segments_per_review = []
            for comment_part in comment_parts:
                if 0 <= index < len(review) and review[index]:
                    segments_per_review.append([comment_part, review[index]])
                index += 1

            segment_aspect_pairs = segment_aspect_pairs + segments_per_review
            for sg in segment_aspect_pairs:
                sentences = self.split_text_into_insentence(sg[0])
                if len(sentences) == 1:
                    temp = sg[1].split(' ')
                    if sg[1] == 'noise':
                        sentiment = 'neutral'
                    else:
                        sentiment = temp[-1]
                        del temp[-1]
                    aspect = " ".join(temp)
                    segment = self.clean_up_text(sentences[0])
                    if segment not in segments:
                        segments.append(segment)
                        aspects.append(aspect)
                        sentiments.append(sentiment)
        data = {
            'segments': segments,
            'aspects': aspects,
            'sentiments': sentiments
        }

        return data

    def clean_up_text(self, sentence):
        cleaned = re.sub(r'^( )+|^[^A-Za-z]+|\.\.+|\,\,+|(_x0085_)+|(-rrb)+|(%)|[^a-zA-Z0-9]+$', r'', sentence)
        return cleaned

    def normalise_aspect_classes(self, aspects):
        staff_attitude_and_professionalism_group = ['staff attitude and professionalism', 'communication']
        care_quality_group = ['care quality', 'process', 'waiting time']
        environment_group = ['food', 'environment', 'resource', 'parking']
        # other_group = ['noise', 'other']
        new_aspects = []
        for aspect in aspects:
            if aspect in staff_attitude_and_professionalism_group:
                new_aspects.append('staff attitude and professionalism')
            elif aspect in care_quality_group:
                new_aspects.append('care quality')
            elif aspect in environment_group:
                new_aspects.append('environment')
            else:
                new_aspects.append('other')

        return new_aspects

    def ngrams(self, sentence, n):
        tokens = sentence.split(' ')
        output = {}
        for i in range(len(tokens) - n + 1):
            g = ' '.join(tokens[i:i + n])
            output.setdefault(g, 0)
            output[g] += 1
        return output

    def get_grouped_aspects(self, training_dataset_file_path):
        data = self.get_segments_aspects_sentiments(training_dataset_file_path)
        segments = data['segments']
        aspects = data['aspects']

        grouped_by_aspects = {}
        for index in range(0, len(segments)):
            if aspects[index] in grouped_by_aspects.keys() and segments[index] not in grouped_by_aspects[aspects[index]]:
                grouped_by_aspects[aspects[index]].append(segments[index])
            elif aspects[index] not in grouped_by_aspects.keys():
                grouped_by_aspects[aspects[index]] = [segments[index]]

        return grouped_by_aspects

    def get_ngrams(self, sent):
        sent = re.sub(r'[^A-Za-z ]+', r'', sent.lower().decode('utf-8'))
        words = [tok for tok in nltk.word_tokenize(sent) if tok not in set(stopwords.words('english')) and len(tok) > 2]
        my_bigrams = ['_'.join(gram) for gram in list(ngrams(words, 2))]
        my_trigrams = ['_'.join(gram) for gram in list(ngrams(words, 3))]

        return words + my_bigrams + my_trigrams

    def convert_list_to_utf8(self, data):
        converted_data = data
        if len(data)>0 and isinstance(data[0], str):
            converted_data = [segment.decode('utf-8', 'ignore') for segment in data]
        return converted_data

    def save_list_as_csv(self, data_list, file_path):
        with open(file_path, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(data_list)