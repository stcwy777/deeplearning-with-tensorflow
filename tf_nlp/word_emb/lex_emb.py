"""Calculate Word Embeddings

This module defines classes and API to calculate word embedding w/t lexical and
semantic constraints. Two algorithms CBOW and SKIPGRAM are implemented as basic
word embedding training frameworks. Joint RCM model is designed to train word
embeddings from relational data (pairwise data).
This module is built upon TensorFlow

A brief reference of the algorithms implemented in this module
T. Mikolov, J. Dean "Distributed representations ... compositionality." (2013)
M. Yu, M. Dredze "Improving Lexical Embeddings with Semantic Knowledge." (2014)

"""
# Python default
import os
import random
import string
# Third party
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
# Inner reference
from tf_nlp.proc_corpus.corpus import Corpus
from tf_nlp.cfg import *

# TODO: evaluate graph; relation embedding; multi threads training...

__author__ = 'yunwang@us.ibm.com (Yun Wang)'


class LexWord2Vec(object):
    """Learn word embeddings by incorporating lexical constraints. Current
    version supports CBOW, SKIPGRAM and RCM (relation constraints model).

    Attributes:
        ====================Data Related====================
        _seq_idx: index of current batch-retrieving from seq data
        _pw_idx: index of current batch-retrieving from pw data
        seq_data: sequential data of word indices
        pw_data: pairwise data of word indices
        idx2word: a dict of {index: word}
        vocab_counts: word frequency extracted from raw corpus
        vocab_size: number of words in the raw corpus
        words_per_epoch: words that need to train in an epoch
        epoch_to_train: # of epochs for training
        trained_words: # of already trained words
        word2idx: a dict of {word: id}
        =================TensorFlow Related=================
        _tf_graph: TF graph object (define the NN structure)
        _sess: TF session object (control the TF tasks)
        _seq_inputs: a tensor to receive batch inputs from seq_data
        _pw_inputs: a tensor to receive batch inputs from qw_data
        _train_labels: a tensor to receive true labels same for seq & pw
        _train_prog: a tensor records training progress
        _emb: word embeddings
        _dyn_eta: a tensor to calculate decaying learning rate
        _loss: a tensor to calculate loss function
        _opt: a tensor to optimize the loss function
        ===================Model Related===================
        context_size: the number of words to predict to the left and right
            of the target.
        emd_dim: # of dimensions of embeddings
        batch_size: # of training instances in a mini-batch
        neg_size: # of negative sampling instances
        eta: default (starting point) of learning rate
        rcm_ratio: x means train rcm one step every x regular training steps
    Methods:

    """

    def __init__(self, model_path, word2idx, context_size=2, emb_dim=300,
                 eps=1e-04, batch_size=128, neg_size=16, model='cbow',
                 valid_size=20, rcm_ratio=0):
        """Constructor. Setup key parameters for training word embedding.

        Args:
            model_path: a path used to save trained model
            word2idx: a dict of {word: id}
            context_size: the number of words to predict to the left and right
                          of the target.
            eta: default learning rate
            emb_dim: size of word embeddings
            batch_size: size of mini-batch
            neg_size: size of negative samplers
            model: default is skip-gram. possible: cbow, skip-gram, pairwise
            valid_size: # of top frequent words for validation (kNN)
        """
        # Data related (will be initialized in training phase)
        self._seq_idx = 0
        self.seq_data = None
        self._pw_idx = 0
        self.pw_data = None
        self.idx2word = None
        self.vocab_counts = None
        self.vocab_size = None
        self.words_per_epoch = 0
        self.epoch_to_train = 0
        self.trained_words = 0
        self.word2idx = word2idx

        # Setup tensors in TF
        self._tf_graph = tf.Graph()
        self._sess = None
        self._train_inputs = None
        self._train_labels = None
        self._train_prog = None
        self._emb = None
        self._sm_w_t = None
        self._sm_b = None
        self._dyn_eta = None
        self._loss = None
        self._opt = None
        self._g_steps = None

        # Word embedding model parameters
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.neg_size = neg_size
        # self.eta = eta
        self.eps = eps
        self.rcm_ratio = rcm_ratio
        self.model_name = model
        self.model_folder = model_path
        self.model_save_path = os.path.join(model_path, model)
        self.valid_size = valid_size
        self.valid_samples = []

    def _init_graph(self):
        """Initialize TF graph in which each operation is a conceptually node
        Model specifications
        1) Word embeddings are initialized with a value between [-1.0, 1.0]
        2) CBOW: Input vector is the average of context words in each dimension
        3) All computation steps defined in this function will be execute during
        the training session.
        """
        with self._tf_graph.as_default():
            # Global step: scalar, i.e., shape [].
            self._g_steps = tf.Variable(0, name="global_step")

            # Initialize embeddings: [vocab_size, emb_dim]
            init_width = 0.5 / self.emb_dim
            emb = tf.Variable(tf.random_uniform([self.vocab_size, self.emb_dim],
                                                -init_width, init_width),
                              name="emb")
            self._emb = emb

            # Softmax weight: [vocab_size, emb_dim]. Transposed.
            self._sm_w_t = tf.Variable(tf.zeros([self.vocab_size,
                                                 self.emb_dim]),
                                       name="sm_w_t")
            # Softmax bias: [emb_dim].
            self._sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

            # Setup batch tensors and batch generation function for skip-gram
            # or CBOW depends on user specification. For CBOW, the input will be
            # an average of all context words
            if self.model_name == 'skip-gram':
                self._train_inputs = tf.placeholder(tf.int64,
                                                    shape=[self.batch_size])
                inputs_emb = tf.nn.embedding_lookup(self._emb,
                                                    self._train_inputs)
                self.gen_batch = self.gen_batch_skip_gram
            elif self.model_name == 'cbow':
                self._train_inputs = tf.placeholder(tf.float32,
                                                    shape=[self.batch_size,
                                                           self.emb_dim])
                self._ctx_inputs = tf.placeholder(tf.int64,
                                                  shape=[None])
                self._ctx_emb = tf.nn.embedding_lookup(self._emb,
                                                       self._ctx_inputs)
                self._ctx_avg = tf.reduce_mean(self._ctx_emb, 0)

                inputs_emb = self._train_inputs
                self.gen_batch = self.gen_batch_cbow

            # In this implementation: labels are word indices
            self._train_labels = tf.placeholder(tf.int64,
                                                shape=[self.batch_size, 1])

            # Use noise-contrastive estimation as loss func
            # Option 1:
            # true_logits, sampled_logits = self._forward_nn(inputs_emb)
            # self._loss = self._nce_loss(true_logits, sampled_logits)

            # Option 2:
            self._loss = tf.reduce_mean(tf.nn.nce_loss(
                self._sm_w_t, self._sm_b, inputs_emb,
                self._train_labels, self.neg_size, self.vocab_size))

            # Add tensor node for optimization
            self._optimize(self._loss)
            # Add accumulated loss for performance monitoring
            self.acc_loss = tf.Variable(0.0, dtype=tf.float32, name='acc_loss',
                                        trainable=True)
            self.acc_loss_op = self.acc_loss.assign_add(self._loss)

            # Normalize word embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(self._emb),
                                         1, keep_dims=True))
            self.norm_emb = tf.truediv(self._emb, norm)

            # Evaluation tensors: convert a list of ids into a tensor
            stop_idx = [self.word2idx[w] for w
                        in set(stopwords.words('english'))
                        if w in self.word2idx]
            cands = set([x for x in xrange(self.vocab_size)
                         if x not in stop_idx])
            self.valid_samples = random.sample(cands, self.valid_size)
            self.valid_data = tf.constant(self.valid_samples)

            # Retrieve normalized embeddings for validation words
            self.valid_emb = tf.nn.embedding_lookup(self.norm_emb,
                                                    self.valid_data)
            self.cos_sim = tf.matmul(self.valid_emb, self.norm_emb,
                                     transpose_b=True)
            # Add model saver
            self.saver = tf.train.Saver()
            # Initialize all the TF tensors
            self.init = tf.initialize_all_variables()

        # Initialize TF session using the graph that created above
        self._sess = tf.Session(graph=self._tf_graph)
        # config=tf.ConfigProto(device_count={'GPU':0}))

    def _forward_nn(self, inputs):
        """Build the feed forward NN given inputs
        Args:
            inputs: inputs to the feed forward NN
        Returns:
            true_logits: logits of inputs
            sampled_logits: logits of negative samples
        """
        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(tf.cast(self._train_labels,
                                           dtype=tf.int64),
                                   [self.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.neg_size,
            unique=True,
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=self.vocab_counts))

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(self._sm_w_t, self._train_labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(self._sm_b, self._train_labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(self._sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(self._sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(inputs, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # Replicate sampled noise lables for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.neg_size])
        sampled_logits = tf.matmul(inputs,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    @staticmethod
    def _nce_loss(true_logits, sampled_logits):
        """Build the graph for the NCE loss.
        Args:
            true_logits: logits of inputs
            sampled_logits: logits of negative samples
        Returns:
            nce_loss_tensor: a TF tensor to calculate nce loss
        """
        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits, tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits, tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.

        nce_loss_tensor = (tf.reduce_mean(true_xent) +
                           tf.reduce_mean(sampled_xent)) #/ self.batch_size
        return nce_loss_tensor

    def _optimize(self, loss):
        """Build the graph to optimize the loss function.
        Args:
            loss: a TF tensor of loss function
        """

        # Optimizer nodes.
        # Linear learning rate decay.
        self._train_prog = tf.placeholder(tf.float32)
        words_to_train = float(self.words_per_epoch * self.epoch_to_train)
        # self._dyn_eta = self.eta * tf.maximum(
        #     0.0001, 1.0 - self._train_prog / words_to_train)
        # optimizer = tf.train.GradientDescentOptimizer(self._dyn_eta)
        optimizer = tf.train.AdamOptimizer(self.eps)
        self._opt = optimizer.minimize(loss,
                                       global_step=self._g_steps,
                                       gate_gradients=optimizer.GATE_NONE)

    def eval_knn(self, top_k=5, use_filter=True):
        """Evaluate word embeddings by displaying k nearest neighbours

        Args:
            top_k: how many word embeddings to evaluate
            use_filter: (boolean) filter out stopwords and punctuations
        """
        assert top_k <= self.valid_size

        # Obtain similarities (cosine similarity)
        sim = self._sess.run(self.cos_sim)
        valid_words = []

        for i in xrange(self.valid_size):
            valid_word = self.idx2word[self.valid_samples[i]]
            if valid_word in string.punctuation or \
               valid_word in set(stopwords.words('english')):
                continue

            # Record valid words and obtain its nearest neighbors
            valid_words.append(valid_word)
            nearest = (-sim[i, :]).argsort()[1:]
            near_words = []

            # Also filter out stopwords in neighbors if use_filter is set
            for word_indx in nearest:
                if use_filter:
                    if self.idx2word[word_indx] in string.punctuation or \
                                    self.idx2word[word_indx] in \
                                    set(stopwords.words('english')):
                        continue
                near_words.append(self.idx2word[word_indx])
                if len(near_words) >= top_k:
                    break

            print "Nearest to '%s': %s" % (valid_word, ','.join(near_words))

    def gen_batch_skip_gram(self):
        """Generate batch for training SKIPGRAM

        Returns:
            batch: a flatten matrix of size batch_size of target word index
            label: a matrix of shape [batch_size, 1] of context words ids
        """
        skip_size = 2 * self.context_size
        # Batch size must be divided evenly by skip size
        assert self.batch_size % skip_size == 0

        # For skip-gram, the input batch is a flatten array of target words
        batch = np.ndarray(shape=[self.batch_size], dtype=np.int64)
        # Labels are ids of the context words of target words
        labels = np.ndarray(shape=[self.batch_size, 1], dtype=np.int64)

        # Raw text data and its size are re-used many times, so setup the two
        # variables to make codes clean
        data = self.seq_data
        data_size = len(data)

        # A span has the target word in the middle. All surrounding words are
        # context of the target. Generate mini-batch by rotating a window of
        # span_size on the input data. (done by the self._seq_idx variable)
        span_size = 2 * self.context_size + 1
        num_spans = self.batch_size / skip_size
        for i in xrange(int(num_spans)):
            tar_word = data[(self._seq_idx + self.context_size) % data_size]

            ctx_idx = [(x + self._seq_idx) % data_size for x in range(span_size)
                       if x != self.context_size]
            random.shuffle(ctx_idx)

            for j in xrange(skip_size):
                batch[i * skip_size + j] = tar_word
                labels[i * skip_size + j, 0] = data[ctx_idx[j]]
            self.trained_words += 1
            self._seq_idx = (self._seq_idx + 1) % data_size

        return batch, labels

    def gen_batch_cbow(self):
        """Generate batch for training CBOW

        Returns:
            batch: a matrix of shape [batch_size, emb_dim] of context embedding
            label: a matrix of shape [batch_size, 1] of target word index
        """
        # Batch size must be large enough for one CBOW instance
        assert self.batch_size > self.context_size * 2

        # For CBOW batch input is mean of context embs [batch_size, emd_dim]
        batch = np.ndarray(shape=[self.batch_size, self.emb_dim],
                           dtype=np.float32)
        # Labels are defined the same as SKIPGRAM
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int64)

        data = self.seq_data
        data_size = len(data)

        # In CBOW, labels are ids of target words. Two variables l_bound and
        # r_bound are used to identify the range of context words
        for i in xrange(self.batch_size):
            tar_word = data[(i + self._seq_idx) % data_size]
            labels[i, 0] = tar_word

            l_bound = range(i - self.context_size, i)
            r_bound = range(i + 1, i + 1 + self.context_size)
            ctx_words = [data[(x+self._seq_idx) % data_size]
                         for x in l_bound + r_bound]
            ctx_avg = self._sess.run(self._ctx_avg,
                                     feed_dict={
                                         self._ctx_inputs: ctx_words
                                     })
            batch[i] = ctx_avg
            self.trained_words += 1
            self._seq_idx = (self._seq_idx + 1) % len(self.seq_data)
        return batch, labels

    def gen_batch_rcm(self):
        """Generate batch for training Relation Constrained Model (RCM). Batch
        generation method depends on the NN structure (CBOW or SKIPGRAM).

        Returns:
            batch: a matrix of context (flatten array of batch_size)
            label: a matrix of shape [batch_size, 1] of target word index
        """
        # Batch size must be exactly divided by 2
        assert self.batch_size % 2 == 0
        pw_data = self.pw_data
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int64)

        # If the training algorithm is SKIPGRAM, two words in a pair will be the
        # context of each other. Use variable 'count'to determine if
        # enough batch data has been generated.
        if self.model_name == 'skip-gram':
            batch = np.ndarray(shape=[self.batch_size], dtype=np.int64)
            count = 0
            while count < self.batch_size:
                # Use every paired word to predict target word label
                tar_word = pw_data[self._pw_idx][0]
                self.trained_words += 1
                for paired_word in pw_data[self._pw_idx][1]:
                    batch[count] = paired_word
                    labels[count, 0] = tar_word
                    batch[count + 1] = tar_word
                    labels[count + 1, 0] = paired_word
                    count += 2
                    if count >= self.batch_size:
                        break
                self._pw_idx = (self._pw_idx + 1) % len(pw_data)
        elif self.model_name == 'cbow':
            batch = np.ndarray(shape=[self.batch_size, self.emb_dim],
                               dtype=np.float32)
            count = 0
            while count < self.batch_size:
                # paired_words is a list of words paired with tar_word.
                # pos_ctx is all possible combinations of paired_words. Each
                # combination will be a context of tar_word in CBOW
                tar_word = pw_data[self._pw_idx][0]
                paired_words = pw_data[self._pw_idx][1]
                # print tar_word, paired_words
                ctx_avg = self._sess.run(self._ctx_avg,
                                         feed_dict={
                                             self._ctx_inputs: paired_words
                                         })
                batch[count] = ctx_avg
                labels[count, 0] = tar_word
                self.trained_words += 1
                count += 1
                self._pw_idx = (self._pw_idx + 1) % len(pw_data)

        return batch, labels

    def end_session(self):
        """ Close session and clean data in the TF
        """
        self._sess.close()

    def train(self, data, pw_data, vocab_size, rev_vocab, counts, epoch=3,
              ck_point=2000, report=200):
        """Train the word embeddings

        Args:
            data: a sequence of integers converted from original text
            pw_data: training data from pairwise input for CBOW
            vocab_size: number of words in the vocabulary
            rev_vocab: a dict of {index: word}
            counts: word frequency in corpus
            epoch: # of epoch to train
            ck_step: # of steps of checkpoint (save the model)
        Returns:
            final_emb: trained word embeddings
        """
        self.seq_data = data
        self.idx2word = rev_vocab
        self.vocab_size = vocab_size
        self._seq_idx = 0
        self._pw_idx = 0
        self.pw_data = pw_data
        self.vocab_counts = counts
        self.epoch_to_train = epoch
        self.words_per_epoch = len(data) + len(pw_data)
        self.trained_words = 0
        epoch = 0
        step = 1
        rcm_step = 0

        # Call the initialization func and fire it in a TF session
        self._init_graph()
        self._sess.run(self.init)

        print("========Start Training Word Embeddings========")
        # Attempt to read saved check point
        ckpt = tf.train.get_checkpoint_state(self.model_folder)
        if ckpt and ckpt.model_checkpoint_path:
            # Load saved session
            self.saver.restore(self._sess, ckpt.model_checkpoint_path)

        # Each step starts with a gen_batch call, then feed the mini-batch in a
        # feed_dict. TF session will then execute one step of training, which is
        # essentially feeding a mini-batch and update gradient and loss function
        while epoch < self.epoch_to_train:
            # If seq_data is available, train the regular CBOW or SKIPGRAM
            if len(data) > 0 and self.rcm_ratio != ONLY_RCM \
                    and ((self.rcm_ratio != NO_RCM
                         and step % self.rcm_ratio) != 0
                         or self.rcm_ratio == NO_RCM):
                batch_inputs, batch_labels = self.gen_batch()
            # If pairwise (relation) data is available, train rcm under two
            # conditions: 1) no seq data 2) train one step rcm every x regular
            # training on seq data
            elif len(pw_data) > 0 and self.rcm_ratio != NO_RCM:
                if (self.rcm_ratio == ONLY_RCM or len(data) == 0) \
                        or (step % self.rcm_ratio == 0):
                    batch_inputs, batch_labels = self.gen_batch_rcm()
                    rcm_step += 1
            else:
                raise IOError("No data available for training.")
            feed = {self._train_inputs: batch_inputs,
                    self._train_labels: batch_labels,
                    self._train_prog: self.trained_words}
            _, loss, acc_loss, g_steps = self._sess.run([self._opt,
                                                         self._loss,
                                                         self.acc_loss_op,
                                                         self._g_steps],
                                                        feed_dict=feed)
            epoch = int(self.trained_words / self.words_per_epoch)
            avg_loss = acc_loss / g_steps
            if step % report == 0 and step:
                # print np.sum(a), np.sum(b), np.max(a), np.max(b)
                print "Current loss at step %d (global: %d rcm steps: %d): " \
                      "%.3f (avg loss: %.3f) epoch:%d" % \
                        (step, g_steps, rcm_step, loss, avg_loss, epoch)
            step += 1
            # Check performance every ck_step
            if g_steps % ck_point == 0 and g_steps:
                # Save the model to a local location
                self.saver.save(self._sess, self.model_save_path,
                                global_step=g_steps)
                print "Model %d saved at %s" % (g_steps,
                                                self.model_save_path)
        # Retrieve the final embeddings from the TF graph
        final_emb = self._sess.run(self.norm_emb)

        return final_emb


def test_module():
    model = Corpus()
    max_ctx, vocab_size = model.load_by_pairs('./data/kg_synonyms_pairs.csv')
    conv_data, pw_data, vocab, rev_vocab = model.build_vocab(vocab_size)
    print max_ctx, vocab_size, len(vocab)
    word2vec = LexWord2Vec('', model='cbow')
    # word2vec.train(conv_data, ctx_word, len(vocab), rev_vocab)
    batch, label = word2vec.get_batch_pairs()

    print [rev_vocab[x] for x in batch[0]]
    print rev_vocab[label[0][0]]

if __name__ == '__main__':
    # with ChangeDirectory(DATA_DIR):
    test_module()
