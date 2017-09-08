from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser
from pycorenlp import StanfordCoreNLP
from utilities import Utilities
import requests
import re
import os
import sys


class Stanford:
    def __init__(self):
        self.stanford_cnlp_url = "http://localhost:9000"
        self.stanford_cnlp_tregex_url = "http://localhost:9000/tregex"
        self.path_to_stanford_models = os.path.expanduser('resources/stanford-parser-full-2016-10-31/')
        self.stanford_cnlp = StanfordCoreNLP(self.stanford_cnlp_url)
        self.utilities = Utilities()

    def _call_stanford(self, annotators, sentence):
        stanford_properties = {
            'annotators': annotators,
            'outputFormat': 'json',
        }

        results = []
        try:
            reload(sys)
            sys.setdefaultencoding('utf-8')
            results = self.stanford_cnlp.annotate(str(sentence), stanford_properties)
        except Exception:
            print "Sorry, Stanford CoreNLP Server not found. Please run the file 'stanford_server.py'"
            exit()

        return results

    def get_stanford_lemma(self, sentences):
        lemmatized = {}
        tokens = self.tokenize(sentences)
        for token in tokens:
            results = self._call_stanford('lemma', str(token))
            lemmas = []
            for result in results['sentences']:
                lemmas = [lemma["lemma"] for lemma in result["tokens"]]
            lemmatized[token] = "".join(lemmas)

        return lemmatized

    def get_stanford_sentiment(self, sentences):
        results = self._call_stanford('sentiment', sentences)

        sentiment = {}
        if type(results) is dict:
            for result in results['sentences']:
                # Convert coreNLP's five classes sentiment to three classes
                converted_sentiment_value = 1
                if int(result["sentimentValue"]) > 2:
                    converted_sentiment_value = 2
                elif int(result["sentimentValue"]) < 2:
                    converted_sentiment_value = 0

                sentiment = {
                    "sentiment": self.utilities.sentiment_classes[converted_sentiment_value],
                    "sentimentValue": converted_sentiment_value,
                }
        else:
            print "Error: No sentiment found for: " + sentences
            exit()

        return sentiment

    def get_stanford_pos(self, sentences):
        pos_tags = []
        results = self._call_stanford('pos,lemma', str(sentences))

        for result in results['sentences']:
            words = [token['word'].lower() for token in result['tokens']]
            pos_token = [token['pos'] for token in result['tokens']]
            pos_tags = dict(zip(words, pos_token))
            break

        return pos_tags

    def _get_string_from_tree(self, str_tree):
        phrase = ""
        pattern = '\([A-Z$]{2,4}[ ][a-zA-Z0-9]*\)'
        reg = re.compile(pattern)

        if reg.match(str_tree):
            target = re.sub("^\([A-Z$]{2,4}[ ]", "", str_tree)
            target = re.sub("\)$", "", target)
            phrase = phrase + self._get_string_from_tree(target)
        else:
            reg2 = re.compile('\([A-Z$]{2,4}[ ][a-zA-Z0-9]*\)')
            if reg2.search(str_tree):
                iterator = reg2.finditer(str_tree)
                for match in iterator:
                    position_range = match.span()
                    subtoken = str_tree[position_range[0]:position_range[1]]
                    # print subtoken
                    phrase = phrase + self._get_string_from_tree(subtoken)
            else:
                phrase = phrase + " " + str_tree

        return phrase

    # TODO: use this function for lexicon generation
    def get_noun_phrases(self, sentence):
        pattern = "(NP[$VP]>S)|(NP[$VP]>S\\n)|(NP\\n[$VP]>S)|(NP\\n[$VP]>S\\n)"
        phrases = self.get_phrases_by_pattern(pattern, sentence)

        return phrases

    def get_phrases_by_pattern(self, pattern, sentence):
        request_params = {"pattern": pattern}
        phrases_data = requests.post(self.stanford_cnlp_tregex_url, data=sentence, params=request_params)

        data = phrases_data.json()
        # for phrase in data:
        phrases_of_sentences = data['sentences']

        noun_phrases = []

        for phrases_of_sentence in phrases_of_sentences:

            for i in range(0, len(phrases_of_sentence)):
                tree_of_phrase = phrases_of_sentence[str(i)]['match']
                phrase = self._get_string_from_tree(tree_of_phrase).strip()
                noun_phrases.append(phrase)

        return noun_phrases

    def get_words_by_pos(self, sentence, pos_type):
        selected_words = {}
        tokens = self.get_stanford_pos(sentence)

        for word, pos in tokens.iteritems():
            if pos_type == 'noun' and pos in self.utilities.noun_phrase_tags:
                selected_words[word] = pos
            elif pos_type == 'verb' and pos in self.utilities.verb_phrase_tags:
                selected_words[word] = pos
            elif pos_type == 'adjective' and pos in self.utilities.verb_phrase_tags:
                selected_words[word] = pos
            elif pos_type == 'adverb' and pos in self.utilities.verb_phrase_tags:
                selected_words[word] = pos
        return selected_words

    def get_phrase_tree(self, sentence):
        results = self._call_stanford('parse', str(sentence))

        return results['sentences'][0]['parse']

    # TODO: incomplete method
    def get_clauses(self, sentence):
        # pattern = "(NP[$VP]>S)|(NP[$VP]>S\\n)|(NP\\n[$VP]>S)|(NP\\n[$VP]>S\\n)"
        pattern = "(S>S)|SBAR<S"
        phrases = self.get_phrases_by_pattern(pattern, sentence)

        return phrases

    def tokenize(self, sentence):
        results = self._call_stanford('pos', sentence)
        sentence_infos = results['sentences']
        tokens = []
        for sentence_info in sentence_infos:
            tokens = tokens + [token['word'].lower() for token in sentence_info['tokens']]

        return tokens

    def _get_stanford_dependency_parsing(self, path_to_stanford_models, text):
        sentences = self.utilities.split_text_into_insentence(text)
        parser = StanfordDependencyParser(path_to_stanford_models + 'stanford-parser.jar',
                                          path_to_stanford_models + 'stanford-parser-3.7.0-models.jar')
        result = parser.raw_parse_sents(sentences)

        return result

    def get_stanford_dependencies(self, sentence):
        result = self._get_stanford_dependency_parsing(self.path_to_stanford_models, sentence)
        dependencies = result.next().next().triples()
        sentence_dependencies = [dependency for dependency in dependencies]

        return sentence_dependencies

    def get_stanford_phrased_structure(self, text):
        sentences = self.utilities.split_text_into_insentence(text)
        parser = StanfordParser(self.path_to_stanford_models + 'stanford-parser.jar',
                                self.path_to_stanford_models + 'stanford-parser-3.7.0-models.jar')
        result = parser.raw_parse_sents(sentences)

        return result

    def get_named_entities(self, sentence):
        results = self._call_stanford('ner', str(sentence))
        sentence_infos = results['sentences']
        named_entities = {}
        for sentence_info in sentence_infos:
            ner_parts = []
            ne = None
            for index, ner_token in enumerate(sentence_info['tokens']):
                if ner_token['ner'] == 'O':
                    if ner_parts:
                        ner_phrase = " ".join(ner_parts)
                        named_entities[ner_phrase] = ne
                    ner_parts = []
                else:
                    ner_parts.append(ner_token['word'])
                    ne = ner_token['ner']

                    if index == len(sentence_info['tokens'])-1:
                        if ner_parts:
                            ner_phrase = " ".join(ner_parts)
                            named_entities[ner_phrase] = ne

        return named_entities

