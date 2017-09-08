import os
import warnings
import pickle
from nltk.stem import WordNetLemmatizer
from openpyxl import load_workbook
from segmenter import Segmenter
from stanford import Stanford
from utilities import Utilities
from wrapper_classifiers import AspectClassifier, SentimentClassifier

import pandas as pd

class Processor(object):

    def __init__(self, settings=None):
        self.settings = settings
        self.utilities = Utilities()
        self.stanford = Stanford()
        self.segmenter = self.load_segmenter()
        self.wordnet_lemmatizer = WordNetLemmatizer()

        self.ml_asp_classifier = AspectClassifier(casecade=False)
        if settings is not None:
            model_path = settings['training_file']+'.pickle'
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
            self.ml_snt_classifier.train(settings['training_file'])

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
                aspects = self.get_aspect_for_segments(segments, 'ml')
                sentiments = self.get_sentiment_for_aspects(segments, mode='ml')
            else:
                # Assign 'other' for noisy reviews to keep indexes same
                aspects = ['other']
                sentiments = ['negative']

            if len(segments) == 1:
                other_words = ['excellent', 'good', 'very good', 'bad', 'ok', 'no response']
                if segments[0] in other_words or len(self.stanford.tokenize(segments[0])) == 1:
                    aspects = ['other']

            # Post-processing: remove consecutive duplicate aspects
            asp_snt_pair = []
            for i, aspect in enumerate(aspects):
                if i > 0 and aspect == aspects[i - 1] and sentiments[i] == sentiments[i - 1]:
                    continue
                else:
                    asp_snt_pair.append(aspect + ' ' + sentiments[i])

            result = [reviews[index]] + list(set(asp_snt_pair))
            reviews_result.append(result)

        self.utilities.save_list_as_csv(reviews_result, output_file)
        print ("Output saved to %s" % output_file)


    def get_aspect_for_segments(self, segments, mode='ml'):
        aspects = []
        if mode == 'rule':
            for segment in segments:
                aspects.append(self.get_aspect_for_segment(segment))
        else:
            aspects = self.ml_asp_classifier.predict(segments)

        return aspects

    def get_sentiment_for_aspects(self, segments, mode='ml'):
        sentiments = []

        # if mode == 'stanford':
        #     for segment in segments:
        #         sentiments.append(self.stanford.get_stanford_sentiment(segment)['sentiment'])
        # elif mode == 'sentistrength':
        #     sentistrength = SentiStrength()
        #     sentiments = sentistrength.get_sentiment_for_segments(segments)
        # elif mode == 'sentiment140':
        #     sentiments = self.sentiment_140.get_sentiment_for_segments(segments)
        # else:
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

    def read_reviews_from_excel(self):
        work_book = load_workbook(filename=self.review_excel_file_path)
        sheet_names = work_book.get_sheet_names()

        text_file = open(self.review_output_file_path, "w")
        for sheet_name in sheet_names:
            work_sheet = work_book.get_sheet_by_name(sheet_name)
            for row in work_sheet:
                for cell in row:
                    if cell.column == 'G':
                        try:
                            if cell.value != 'Patient Comments':
                                comment = cell.value + '\n'
                                text_file.write(comment.encode('ascii', 'ignore'))
                        except TypeError:
                            break
        text_file.close()

    def get_all_reviews(self):
        all_reviews = []
        try:
            all_reviews = self.utilities.get_lines_from_text_file(self.review_output_file_path)
        except EnvironmentError:
            self.read_reviews_from_excel()
            all_reviews = self.utilities.get_lines_from_text_file(self.review_output_file_path)
        return all_reviews

    def wordnet_lemmatizing(self, word):
        if not word:
            return ""
        return self.wordnet_lemmatizer.lemmatize(word)

    def get_aspect_for_segment(self, segment):
        aspect = None
        sentence_dependencies = self.stanford.get_stanford_dependencies(segment)
        aspect_candidates = self.rules.get_aspects_candidates(segment, sentence_dependencies)

        if aspect_candidates:
            aspect_list = self._get_real_aspects_from_candidates(aspect_candidates)
            aspect = aspect_list[0] if aspect_list else None  # TODO: must need to fix for multiple aspect in segment

        if not aspect:
            noun_phrases = self.stanford.get_words_by_pos(segment, 'noun')
            aspect_list = self._get_real_aspects_from_candidates(noun_phrases)
            aspect = aspect_list[0] if aspect_list else None  # TODO: must need to fix for multiple aspect in segment

        if not aspect:
            tokens = self.stanford.tokenize(segment)
            aspect_list = self._get_real_aspects_from_candidates(tokens)
            aspect = aspect_list[0] if aspect_list else 'other'  # TODO: must need to fix for multiple aspect in segment

        # post processing
        aspect = self.apply_post_processing_rules(segment, aspect)

        return aspect

    def apply_post_processing_rules(self, segment, aspect):
        care_quality_clues = {'nothing to add', 'nothing to say', 'nothing to improve', 'nothing to change', 'nothing to fix','thank'}
        new_aspect = aspect
        if aspect != 'care quality':
            for clue in care_quality_clues:
                if clue in aspect:
                    new_aspect = 'care quality'
        return new_aspect

    def get_aspects_for_sentence(self, sentence):
        segments_info = self.segmenter.get_segments(sentence)
        segments = segments_info['segments']

        segments_with_aspects = []
        for segment in segments:
            aspect = self.get_aspect_for_segment(segment)
            sentiment_info = self.stanford.get_stanford_sentiment(segment)

            if aspect:
                segments_with_aspects.append(self.prepare_aspect_result_unit(segment, aspect, sentiment_info['sentiment']))
            else:
                segments_with_aspects.append(self.prepare_aspect_result_unit(segment, None, sentiment_info['sentiment']))

        return segments_with_aspects

    def _get_real_aspects_from_candidates(self, candidates):
        aspects = []
        for aspect_candidate in candidates:
            aspects_from_lexicon = self.lexicon.get_aspect_by_word(aspect_candidate)
            aspect_names = []
            for aspect_id in aspects_from_lexicon:
                aspect_names.append(self.lexicon.get_aspect_name_by_id(aspect_id))
                aspects = list(set(aspects + aspect_names))

        return aspects

    def prepare_aspect_result_unit(self, segment, aspect, sentiment):
        aspects_unit = {
            'segment': segment,
            'aspect': aspect,
            'sentiment': sentiment
        }

        return aspects_unit

    def save_output(self, review_aspects):
        """
        Save final output to csv file 
        
        :param review_aspects: 
        :return: 
        """
        output = "Review,Segment,Aspect,sentiment,Segment,Aspect,sentiment,Segment,Aspect,sentiment\n"
        for review_item in review_aspects:
            output_list = ["\"" + review_item['review'] + "\""]
            for item in review_item['items']:
                output_list.append("\"" + item['segment'] + "\"")
                output_list.append('/'.join(item['aspects']))
                output_list.append(item['sentiment']['sentiment'])
                # output_list.append("")

            output += ",".join(output_list) + "\n"

        self.utilities.write_content_to_file(self.csv_file_name, output)
