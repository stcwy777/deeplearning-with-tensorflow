"""Corpus Pre-processing

This module defines classes and API to pre-process corpus for NLP study. There
are two major data sources (KnowledgeBase, text corpus). Classes KnowledgeBase
and Corpus are designed to process the two sources.

KnowledgeBase provides functions to read data from RDF and restore extracted
data to Neo4j import CSV.

Corpus provides API to clean raw text corpus, generate vocabulary and
dictionaries to convert word into indices, and vise versa.
"""

# Python default
from cPickle import load, dump
from collections import Counter, defaultdict
import glob
import os
import re
import rdflib
import string
# Third party
import networkx as nx
from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import DanishStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Inner reference
from tf_nlp.cfg import *
from tf_nlp.utils import get_csv_reader

# TODO: evaluate graph; self-test on public data; relation embedding...

__author__ = 'yunwang@us.ibm.com (Yun Wang)'


class KnowledgeBase(object):
    """A wrapper for knowledge base (RDF, NT, etc.) operation. More functions
    will be added when the

    Attributes:
        rdf_graph: an inner graph object for RDF file [public]
        kb_graph: an inner network object that loads from RDF [public]
    """
    def __init__(self):
        """Constructor. Initialize class attributes.
        """
        self.rdf_graph = rdflib.Graph()
        self.kb_graph = nx.DiGraph()

    def read_rdf(self, path, ft='rdf'):
        """Read RDF file from source. Extract nodes and edges information then
        save in a network object

        Args:
            path: path to input RDF file
            ft: source format by default is rdf. Possible values: 'nt', 'rdf'
        Return:
            kb_graph: a network obj is returned for usage outside class
        """
        self.rdf_graph.parse(path, format=ft)

        for subj, pred, obj in self.rdf_graph:

            if subj == '' or obj == '':         # Skip incomplete
                continue

            subj = subj.split('/')[-1]          # use last component after '/'
            pred = pred.split('/')[-1]
            obj = obj.split('/')[-1]

            # Scan every trips and skip KB_LABEL and empty predicate. An edge is
            # then added only if subject is different from object
            if pred == KB_LABEL:
                if not self.kb_graph.has_node(subj):
                    self.kb_graph.add_node(subj, label=obj)
            elif len(pred) > 0:
                if subj == obj:
                    if not self.kb_graph.has_node(subj):
                        self.kb_graph.add_node(subj, label=obj)
                else:
                    self.kb_graph.add_edge(subj, obj, type=pred)

            return self.kb_graph

    def save_network(self, gexf_path, neo4j_node, neo4j_edge):
        """Save extracted network object into a local GEXF file. Generate CSVs
        to import into Neo4j database

        Args:
            gexf_path: path to save GEXF file
            neo4j_node: csv name to save neo4j nodes info
            neo4j_edge: csv name to save neo4j edges info
        """

        nx.write_gexf(self.kb_graph, gexf_path)
        node_ids = {}

        # Read nodes from network obj and save them into CSV. Each line is an
        # node followed by its attribute. In this version, node has its name and
        # a label which is essentially the same as name by used for filtering
        with open(neo4j_node, 'w') as node_output:
            node_output.write('i:id,name,l:label\n')
            for node in self.kb_graph.nodes():
                node_ids[node] = len(node_ids)
                node_f = node.replace('"', '')
                node_f = node_f.replace("'", '')
                node_f = node_f.replace('\n', '')
                node_output.write('%d,"%s","%s"\n' % (node_ids[node],
                                                      node_f,
                                                      'ENTITY'))

        # Read edges from network obj and save them into CSV. Each line has the
        # names of two connected nodes (the sequence of start & end sequence
        # matters for directed graph). Edge has a TYPE attribute which is the
        # predicate in RDF.
        with open(neo4j_edge, 'w') as edge_output:
            edge_output.write(':START_ID,:END_ID,:TYPE\n')
            for src, tar in self.kb_graph.edges_iter():
                edge_output.write('%d,%d,"%s"\n' %
                                  (node_ids[src], node_ids[tar],
                                  self.kb_graph[src][tar]['type']))


class Corpus(object):
    """A wrapper to pre-process text corpus.

    Attributes:
        word2idx: a dict of corpus vocabulary {word: index}
        idx2word: a dict of reversed corpus vocabulary {index: word}
        words: all words from input corpus
        raw_data: raw text input (paraphrases, articles, sentences and etc.)
        seq_data: a list of integers converted from words in corpus
        count: a dict of word frequency in corpus {word: frequency}
        word_ctx: a dict of {word: [a list of paired words]}
        pw_data: a dict of pair-wise data
        docs: a dict of loaded documents {doc_id: corpus}
        lines: a dict of lines in one document {line_id: corpus}
        re_rules: a list of regular expressions to clean corpus.
        ner_tagger / pos_tagger: Stanford NER / POR tagger
        pt_stemmer / dan_stemmer: NLTK stemmer model
        lemma: NLTK lemmatizer based on WordNet
        tfidf: TF-IDF vectorizer for document term index
        lib_path: path to the third party directory
    """
    def __init__(self, lib_path=THIRD_PATY_PATH):
        """Constructor. Initialize class attributes.
        """
        self.word2idx = {}
        self.idx2word = {}
        self.words = []
        self.raw_data = []
        self.seq_data = None
        self.count = None
        self.word_ctx = defaultdict(set)
        self.pw_data = []
        self.docs = defaultdict(dict)
        self.lines = defaultdict(list)
        self.re_rules = []
        self.lib_path = lib_path
        self.ner_tagger = self.set_ner_tagger(NER_MODEL, NER_JAR)
        self.pos_tagger = self.set_pos_tagger(POS_MODEL, POS_JAR)
        self.pt_stemmer = PorterStemmer()
        self.dan_stemmer = DanishStemmer()
        self.lemma = WordNetLemmatizer()
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize,
                                     stop_words='english')

    def set_filt_rule(self, new_rule):
        """Add a regex rules to clean text

        Args:
            new_rule: a regular expression added to the class filters
        """
        self.re_rules.append(re.compile(new_rule))

    def set_ner_tagger(self, model_path=NER_MODEL, jar_path=NER_JAR):
        """Setup path for Standford NER tagger. Default value is configured in
        cfg.py

        Args:
            model_path: path to the trained model
            jar_path: path to the JAR
        """
        model_path = os.path.join(self.lib_path, model_path)
        jar_path = os.path.join(self.lib_path, jar_path)
        if os.path.isfile(model_path) and os.path.isfile(jar_path):
            self.ner_tagger = StanfordNERTagger(model_path, jar_path)
        else:
            raise IOError('Cannot find NER tagging lib')

    def set_pos_tagger(self, model_path=POS_MODEL, jar_path=POS_JAR):
        """Setup path for Standford POS tagger. Default value is configured in
        cfg.py

        Args:
            model_path: path to the trained model
            jar_path: path to the JAR
        """
        model_path = os.path.join(self.lib_path, model_path)
        jar_path = os.path.join(self.lib_path, jar_path)
        if os.path.isfile(model_path) and os.path.isfile(jar_path):
            self.pos_tagger = StanfordPOSTagger(model_path, jar_path)
        else:
            raise IOError('Cannot find POS tagging lib')

    def tag_ner(self, words):
        """Tag name entities from a sequence of words

        Args:
            words: a list of input words
        """
        return self.ner_tagger.tag(words)

    def tag_pos(self, words):
        """Assign POS tag for a sequence of words

        Args:
            words: a list of input words
        """
        return self.pos_tagger.tag(words)

    def stem_tokens(self, tokens, impl='Porter'):
        """Stem tokens using NLTK stem model

        Args:
            tokens: a list of tokens
            impl: stem algorithm (by default 'Porter', possible 'Danish')
        Returns:
            stemmed: a list of stemmed tokens
        """
        stemmed = []
        stem = self.pt_stemmer

        if impl == 'Porter':
            stem = self.dan_stemmer

        for item in tokens:
            stemmed.append(stem.stem(item))
        return stemmed

    def lemma_tokens(self, tokens):
        """Lemmatize tokens using NLTK lemmatization model

        Args:
            tokens: a list of tokens
        Returns:
            lemmatized: a list of lemmatized tokens
        """
        lemmatized = []
        for item in tokens:
            lemmatized.append(self.lemma.lemmatize(item))
        return lemmatized

    @staticmethod
    def norm_num(tokens):
        """Normalize numbers by converting numbers in the tokens into '0'

        Args:
            tokens: a list of tokens may contain numbers
        Returns:
            conv_t: a list of converted tokens
        """
        conv_t = []
        for t in tokens:
            # Removing punctuations from strings first
            exclude = set(string.punctuation)
            new_t = ''.join(ch for ch in t if ch not in exclude)
            try:
                is_num = unicode(new_t, 'utf-8').isnumeric()
            except TypeError:
                is_num = new_t.isnumeric()

            if is_num:
                conv_t.append('0')
            else:
                conv_t.append(t)
        return conv_t

    def tokenize(self, text):
        """Tokenize a piece of text using NLTK tokenizer. This function will be
        used in self.tfidf to calculate TF-IDF value

        Args:
            text: a string obj
        """
        tokens = word_tokenize(text)
        return tokens

    def get_docs_tfidf(self):
        """Calculate TF-IDF from documents using scikit-learn feature extraction
        model. Documents are stored in an inner dictionary by using load_by_docs
        first.

        Returns:
            words: array mapping indices to words
            tfidf_mat: Tf-idf matrix.
        """
        words = self.tfidf.get_feature_names()
        tfidf_mat = self.tfidf.fit_transform(self.docs.values())
        return words, tfidf_mat

    def load_by_lines(self, path, use_filter=True, use_stem=True,
                      use_lemma=True, add_eos=True, add_eot=False,
                      conv_num=True, store_words=True):
        """Load sequences of words from a document by lines. Apply proper
        pre-processing based on parameters. By default, one line is considered
        as a sentence

        Args:
            path: a path to the input file
            use_filter: use preset regular expressions
            use_stem: use preset stemmer
            use_lemma: use preset lemmatizer
            add_eos: add EOS label to the end of each line
            add_eot: add EOT label to where indicate an end of turn
            conv_num: normalize numbers. (see norm_num function)
            store_words: store all words in a inner list. Set true for training
                         word embeddings

        Returns:
            lines: a dictionary of processed corpus {line_num: tokens}
        """
        lines = defaultdict(list)
        with open(path, 'r') as input_file:
            for line in input_file:
                # Remove EOL and lower the words
                lowers = line.rstrip().lower()

                # Pre-process based on parameter setting
                if use_filter:
                    for reg in self.re_rules:
                        lowers = re.sub(reg, '', lowers)

                tokens = self.tokenize(lowers)

                if use_stem:
                    tokens = self.stem_tokens(tokens)
                if use_lemma:
                    tokens = self.lemma_tokens(tokens)
                if conv_num:
                    tokens = self.norm_num(tokens)
                if add_eos:
                    tokens.append('_eos_')
                if add_eot:
                    tokens.append('_eot_')

                if store_words:
                    self.words.extend(tokens)

                self.raw_data.extend(tokens)
                lines[len(lines)] = tokens
        return lines

    def load_by_docs(self, folder_path):
        """Load all text documents under a given folder into an inner dict.
        index in self.docs starts as 1.

        Args:
            folder_path: a path to a folder
        """

        # Detect all txt files under the folder
        for path in glob.glob(os.path.join(folder_path, '*.txt')):
            print 'read text from %s' % path

            with open(path, 'w') as input_file:
                self.docs[len(self.docs) + 1] = input_file.read()

    def load_by_pairs(self, path):
        """Load word pairs from text file to train pairwise embedding. The input
        file must be a CSV in which every line is a pair of two words.

        Args:
            path: a path to a input file CSV
        Returns:
            max_ctx: maximum length of context (# of paired words)
            vocab_size: vocabulary size of the input file
        """

        with open(path, 'r') as input_file:
            for data in get_csv_reader(input_file):
                l_w = data[0].lower()
                r_w = data[1].lower()
                self.words.extend([l_w, r_w])
                # Given a pair of words, update the their context set for
                # each of them. This information will be used for CBOW model
                self.word_ctx[l_w].add(r_w)
                # self.word_ctx[r_w].add(l_w)
        # The maximum length of context is useful for determining context size
        max_ctx = len(max(self.word_ctx.items(), key=lambda x: len(x[1]))[1])
        vocab_size = len(self.word_ctx)
        return max_ctx, vocab_size

    def build_vocab(self, vocab_size=DEF_VOCAB_SIZE):
        """Generate vocabulary over the extracted words. Convert vocabulary
        into integer spaces by indexing words based on counts

        Args:
            vocab_size: vocabulary size
        Returns:
            seq_data: sequences of integers converted from words
            conv_word_ctx: a dict of {word_integer: a list of context}
            vocab: a dict of word-index pairs {word: index}
            idx2word: a dict reverse word-index pairs {index: word}
        """

        # Setup vocabulary and cut off by frequency
        word_counter = map(list, Counter(self.words).most_common(vocab_size))
        # Add EOS, EOT and UNK symbols to vocabulary
        word_counter.extend([[W_UNK, 0], [W_EOS, 0], [W_EOT, 0]])
        self.word2idx = {}
        self.seq_data = []

        # Set index for vocab ordered by counts
        for word, _ in word_counter:
            self.word2idx[word] = len(self.word2idx)

        # Generate sequence of integers according to the position of words in
        # raw data. Out-of-boundary words are counted as W_UNK
        unk_count = 0
        for word in self.raw_data:
            try:
                index = self.word2idx[word]
            except KeyError:
                index = 0
                unk_count += 1
            self.seq_data.append(index)

        # If pairwise data provided, generate its integer representation
        for word, ctx in self.word_ctx.items():
            self.pw_data.append((self.word2idx[word],
                                 [self.word2idx[w] for w in ctx]))
        word_counter[-3][1] = unk_count

        self.count = [0 for _ in xrange(len(word_counter))]

        for (word, count) in word_counter:
            self.count[self.word2idx[word]] = count
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))

        return self.seq_data, self.pw_data, self.word2idx, \
               self.idx2word, self.count

    def ret_domain_key_words(self, top_k=50):
        """Return a list of top k frequent words from input corpus. Stopwords,
        numbers and punctuations are ignored.

        Args:
            top_k: number of top frequent words to retrieve (default: 50)
        Returns:
            key_words: a list of key words
            vocab_code: a list of integer corresponding to key words
        """
        key_words = []
        key_ids = []
        count = top_k
        for (word, freq) in self.count:
            # Valid words. Skip punctuation, stopwords
            if word not in string.punctuation \
                    and word not in set(stopwords.words('english')):
                # Check if the word is number
                try:
                    word = unicode(word, 'utf-8')
                    is_num = word.isnumeric()
                except TypeError:
                    is_num = word.isnumeric()
                if not is_num:
                    key_words.append([word, freq])
                    key_ids.append(self.word2idx[word])
                    count -= 1
            if count == 0:
                break

        return key_words, key_ids

    def serialize(self, folder_path, protocol='PICKLE', force=True):
        """Serialize inner data into a local directory. By default using pickle.
        Will support JSON if necessary. Serialized files are internally named
        so that the module can load them automatically.

        Args:
            folder_path: a path to a folder
            protocol: protocol for serializing
        """

        if not force and not os.path.isdir(folder_path):
            raise IOError('Folder not exists for serialization')

        elif not os.path.exists(folder_path):
            os.makedirs(folder_path)

        conv_path = os.path.join(folder_path, 'seq_data.dt')
        pairwise_path = os.path.join(folder_path, 'pairwise.dt')
        word2idx_path = os.path.join(folder_path, 'word2idx.dt')
        idx2word_path = os.path.join(folder_path, 'idx2word.dt')
        count_path = os.path.join(folder_path, 'count.dt')

        dump(self.seq_data, open(conv_path, 'wb'))
        dump(self.pw_data, open(pairwise_path, 'wb'))
        dump(self.word2idx, open(word2idx_path, 'wb'))
        dump(self.idx2word, open(idx2word_path, 'wb'))
        dump(self.count, open(count_path, 'wb'))

    def load_serialized(self, folder_path):
        """Load serialized data.

        Args:
            folder_path: a path to the folder of serialized data
        Returns:
            seq_data: sequences of integers converted from words
            count: a list of [word, counts]
            vocab: a dict of word-index pairs {word: index}
            idx2word: a dict reverse word-index pairs {index: word}
        """
        if not os.path.isdir(folder_path):
            raise IOError('No serialized data found')

        conv_path = os.path.join(folder_path, 'seq_data.dt')
        pairwise_path = os.path.join(folder_path, 'pairwise.dt')
        vocab_path = os.path.join(folder_path, 'word2idx.dt')
        idx2word_path = os.path.join(folder_path, 'idx2word.dt')
        count_path = os.path.join(folder_path, 'count.dt')

        self.seq_data = load(open(conv_path, 'rb'))
        self.pw_data = load(open(pairwise_path, 'rb'))
        self.word2idx = load(open(vocab_path, 'rb'))
        self.idx2word = load(open(idx2word_path, 'rb'))
        self.count = load(open(count_path, 'rb'))

        return self.seq_data, self.pw_data, self.word2idx, \
               self.idx2word, self.count


def test_module():
    """Code snippet for testing
    """
    # kb = KnowledgeBase()
    # kb.read_rdf('./data/kg/dialog/geico_web_gh_v5_0406.nt')
    # kb.save_network('./data/kg/geico_graph.gexf', './data/kg/geico_data.csv',
    #                 './data/kg/geico_edge.csv')

    model = Corpus()
    max_ctx, vocab_size = model.load_by_pairs('./data/kg_synonyms_pairs.csv')
    print max_ctx, vocab_size
    _, ctx_word, _, idx2word = model.build_vocab(vocab_size)

    print idx2word[ctx_word[0][0]]
    print [idx2word[x] for x in ctx_word[0][1]]

if __name__ == '__main__':
    test_module()
