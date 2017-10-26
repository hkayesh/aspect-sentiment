import os
import re
import warnings
import pickle
from nltk.stem import WordNetLemmatizer
from openpyxl import load_workbook
from segmenter import Segmenter
from utilities import Utilities
from wrapper_classifiers import AspectClassifier, SentimentClassifier

class Processor(object):

    def __init__(self, settings=None):
        self.settings = settings
        self.utilities = Utilities()
        self.segmenter = self.load_segmenter()
        self.wordnet_lemmatizer = WordNetLemmatizer()

        self.ml_asp_classifier = AspectClassifier(casecade=False)
        if settings is not None:
            model_path = settings['training_file']+'.aspect_model.pickle'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as handle:
                    self.ml_asp_classifier = pickle.load(handle)
            else:
                self.ml_asp_classifier.train(settings['training_file'])
                with open(model_path, 'wb') as f:
                    pickle.dump(self.ml_asp_classifier, f)
                print("Aspect Extraction model written out to {}".format(model_path))

        self.ml_snt_classifier = SentimentClassifier()
        if settings is not None:
            model_path = settings['training_file'] + '.sentiment_model.pickle'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as handle:
                    self.ml_snt_classifier = pickle.load(handle)
            else:
                self.ml_snt_classifier.train(settings['training_file'])
                with open(model_path, 'wb') as f:
                    pickle.dump(self.ml_snt_classifier, f)
                print("Sentiment Detection model written out to {}".format(model_path))

    def run(self):
        settings = self.settings

        data_file = settings['data_file']
        output_file = settings['output_file']

        df = self.utilities.read_from_csv(data_file)

        original_reviews = [row[0] for row in df]

        if 'max_reviews' in settings.keys() and settings['max_reviews'] < len(original_reviews):
            original_reviews = original_reviews[:settings['max_reviews']]

        original_reviews = self.utilities.convert_list_to_utf8(original_reviews)

        cleaned_reviews = []
        empty_review_indexes = []
        for index, review in enumerate(original_reviews):
            cleaned_review = self.utilities.clean_up_text(review.lower())
            if len(cleaned_review) > 2:
                cleaned_reviews.append(cleaned_review)
            else:
                cleaned_reviews.append(review.lower())
                empty_review_indexes.append(index)
        reviews = cleaned_reviews

        reviews_segments = []
        for index, review in enumerate(reviews):
            print index
            if index in empty_review_indexes:
                reviews_segments.append([review])
                continue
            sentences = self.utilities.split_text_into_insentence(review)

            # start: force split exceptionally long (more than 800 chars) sentences
            tmp_sentences = []
            for sentence in sentences:
                if len(sentence) > 800:
                    if '|' in sentence:
                        tmp_sentences = tmp_sentences + sentence.split('|')
                    else:
                        first_part, second_part = sentence[:len(sentence) / 2], sentence[len(sentence) / 2:]
                        tmp_sentences = tmp_sentences + [first_part, second_part]
                else:
                    tmp_sentences.append(sentence)

            sentences = tmp_sentences
            # end: force split exceptionally long (more than 800 chars) sentences

            segments = []
            try:
                for sentence in sentences:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        segment_info = self.segmenter.get_segments(sentence)
                    segments = segments + [sg for sg in segment_info['segments'] if len(sg) > 2]
            except AssertionError:
                # print review
                segments = [review]
            reviews_segments.append(segments)

        reviews_result = []

        for index, segments in enumerate(reviews_segments):

            if index not in empty_review_indexes:
                aspects = self.get_aspect_for_segments(segments)
                sentiments = self.get_sentiment_for_aspects(segments)
            else:
                # Assign 'other' for noisy reviews to keep indexes same
                aspects = ['other']
                sentiments = ['negative']

            #aspects = self.apply_dictionaries(segments, aspects)

            if len(segments) == 1:
                other_words = ['excellent', 'good', 'very good', 'bad', 'ok', 'no response']
                if segments[0] in other_words or len(self.utilities.tokenize(segments[0])) == 1:
                    aspects = ['other']

            # Post-processing: remove consecutive duplicate aspects
            asp_snt_pair = []
            for i, aspect in enumerate(aspects):
                if i > 0 and aspect == aspects[i - 1] and sentiments[i] == sentiments[i - 1]:
                    continue
                else:
                    # removed sentiment from output
                    # asp_snt_pair.append(aspect + ' ' + sentiments[i])
                    asp_snt_pair.append(aspect)

            result = [unicode(reviews[index]).encode("utf-8")] + list(set(asp_snt_pair))
            reviews_result.append(result)

        self.utilities.save_list_as_csv(reviews_result, output_file)
        print ("Output saved to %s" % output_file)

    def get_aspect_for_segments(self, segments):
        aspects = self.ml_asp_classifier.predict(segments)

        return aspects

    def get_sentiment_for_aspects(self, segments):
        sentiments = self.ml_snt_classifier.predict(segments)
        return sentiments

    def load_segmenter(self):
        training_file_name = os.path.splitext(self.settings['training_file'])[0]
        outpath = training_file_name + '.segmenter.pickle'
        segmenter = None
        if os.path.exists(outpath):
            with open(outpath, 'rb') as handle:
                segmenter = pickle.load(handle)
        else:
            if outpath is not None:
                segmenter = Segmenter(self.settings['training_file'])
                with open(outpath, 'wb') as f:
                    pickle.dump(segmenter, f)
                print("Segmenter model written out to {}".format(outpath))

        return segmenter

    def wordnet_lemmatizing(self, word):
        if not word:
            return ""
        return self.wordnet_lemmatizer.lemmatize(word)

    def apply_post_processing_rules(self, segment, aspect):
        care_quality_clues = {'nothing to add', 'nothing to say', 'nothing to improve', 'nothing to change', 'nothing to fix','thank'}
        new_aspect = aspect
        if aspect != 'care quality':
            for clue in care_quality_clues:
                if clue in aspect:
                    new_aspect = 'care quality'
        return new_aspect

    def apply_dictionaries(self, segments, aspects):
        food_lexicon = ['food', 'canteen', 'canten', 'coffee', 'cofee', 'coffe', 'coffee', 'tea', 'drink', 'drinks']
        parking_lexicon = ['car park', 'car-park', 'carpark', 'parking', 'bicycle']

        for index, segment in enumerate(segments):
            all_words = self.utilities.get_lemma(segment)
            lemmatized_words = all_words.values()
            # print segment
            for word in food_lexicon:
                if word in lemmatized_words:
                    aspects[index] = 'food'

            for word in parking_lexicon:
                if word in lemmatized_words:
                    aspects[index] = 'parking'

        return aspects

