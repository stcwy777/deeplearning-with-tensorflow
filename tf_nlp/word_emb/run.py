"""Train Word Embeddings

This is the script for training word embeddings by calling it from the shell.
It relies on Class Corpus to generated vocabulary, word indices and training
data from raw corpus and then uses Class LexWord2Vec to train word embeddings.

Users can specify the interval for saving trained model to a local directory.
By default, training status include average loss will be displayed every 500
training steps.

Usage:
from the project root folder (.../cogn-dialog), type:
    python -m tf_nlp.word_emb.run --help
to see how different parameters feed the script.
"""

# Python default
import os
import glob
from cPickle import dump
# Third party
import tensorflow as tf
# Inner reference
from tf_nlp.proc_corpus.corpus import Corpus
from tf_nlp.word_emb.lex_emb import LexWord2Vec


# TODO: use logger...

__author__ = 'yunwang@us.ibm.com (Yun Wang)'

# Use TF API to conveniently define parameters
tf.flags.DEFINE_integer(
    'batch_size', 80, 'Mini-batch size (default = 128)')

tf.flags.DEFINE_integer(
    'ck_point', '2000',
    'Check point: every x steps to save the model (default: 2000')

tf.flags.DEFINE_integer(
    'context_size', 2, 'How many words to the left and right of the target word'
                       'should be used as context (default = 2)')

tf.flags.DEFINE_integer(
    'emb_dim', 200, 'Dimension of word embedding (default = 200)')

tf.flags.DEFINE_float(
    'eps', 1e-04, 'Adam optimizer epsilon (default = 1e-04)')

tf.flags.DEFINE_integer(
    'epoch', 3, ' (Number of epochs for training default = 3)')

tf.flags.DEFINE_string(
    'input_dir', os.path.abspath('./data'),
    'Input directory containing corpus (default = "./data")')

tf.flags.DEFINE_string(
    'model_dir', os.path.abspath('./model'),
    'Model directory for saving models (default = "./model")')

tf.flags.DEFINE_string(
    'model', 'cbow', 'Use model ("cbow" or "skip-gram" default = "cbow")')

tf.flags.DEFINE_integer(
    'neg_size', 15, 'Negative sampling size (default = 64)')

tf.flags.DEFINE_string(
    'output_dir', os.path.abspath('./embeds'),
    'Output directory for word embeddings (default = "./embeds")')

tf.flags.DEFINE_integer(
    'preload', 0, 'Load pre-processed corpus (yes: 1 no:0 default = 0)')

tf.flags.DEFINE_integer(
    'report', 200, 'Steps to report training progress (default = 200)')

tf.flags.DEFINE_integer(
    'rcm', '0',
    'Ratio of training Relational constraints model.'
    'Possible values: -1: only RCM, 0: skip RCM, x >0: train RCM every x steps'
    '(default = 0 skip rcm)')

tf.flags.DEFINE_string(
    'task_name', 'task',
    'Task name, will be used as saved model default = "task")')

tf.flags.DEFINE_string(
    'third_party', os.path.abspath('./third_party'),
    'Directory of third party libs (default = "./third_party")')

tf.flags.DEFINE_integer(
    'vocab_size', 50000, 'Maximum size of vocabulary (default = 50,000)')

FLAGS = tf.flags.FLAGS


def main():
    corpus_proc = Corpus(FLAGS.third_party)

    # Regex rules that remove html tags
    corpus_proc.set_filt_rule('<.*?>')
    corpus_proc.set_filt_rule('\[|\]|\$')
    corpus_proc.set_filt_rule("``.*?''")

    # Examine existence of key folders
    if not os.path.exists(FLAGS.input_dir):
        raise IOError('Input data directory (%s) not exists!' % FLAGS.input_dir)

    if not os.path.exists(FLAGS.output_dir):
        raise IOError('Output data directory (%s) not exists!' % FLAGS.output_dir)

    if not os.path.exists(FLAGS.model_dir):
        raise IOError('Model directory (%s) not exists!' % FLAGS.model_dir)

    # Generate folder_path as input_dir/task_name
    data_folder_path = os.path.join(FLAGS.input_dir, FLAGS.task_name)
    model_folder_path = os.path.join(FLAGS.model_dir, FLAGS.task_name)

    # Load processed corpus from serialized data
    no_data = False
    if FLAGS.preload == 1:
        print '========Load pre-processed corpus========'
        try:
            print data_folder_path
            seq_data, pw_data, word2idx, idx2word, count = \
                corpus_proc.load_serialized(data_folder_path)
            # if FLAGS.pairwise > 0:
        except IOError:
            # If no serialized data found
            no_data = True
            print "No pre-processed data available!"

    # Process text data from raw and generate vocabulary
    if no_data or FLAGS.preload == 0:
        found_txt = glob.glob(os.path.join(FLAGS.input_dir, '*.txt'))
        found_csv = []
        if FLAGS.rcm != -1:
            for txt_file in found_txt:
                print '========Load corpus:%s========' % txt_file
                corpus_proc.load_by_lines(txt_file, add_eos=False,
                                          add_eot=False, use_stem=False)
            if len(found_txt) == 0:
                print 'WARNING: No text data found.'
        if FLAGS.rcm != 0:
            found_csv = glob.glob(os.path.join(FLAGS.input_dir, '*.csv'))
            for csv_file in found_csv:
                print '========Load pairwise data:%s========' % csv_file
                _, _ = corpus_proc.load_by_pairs(csv_file)
            if len(found_csv) == 0:
                print 'WARNING: No relations data found.'

        if len(found_csv) == 0 and len(found_txt) == 0:
            raise IOError('No input data found from specified data root')

        # Process corpus
        seq_data, pw_data, word2idx, idx2word, count = \
            corpus_proc.build_vocab(FLAGS.vocab_size)

        corpus_proc.serialize(data_folder_path, force=True)

    vocab_size = min(len(idx2word), FLAGS.vocab_size)
    assert vocab_size > 0
    print 'Vocabulary Size:', vocab_size

    # Create a new folder for saved models for this task
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    print '========Initialize TensorFlow Graph========\n' \
          'Model: %s\nContext size: %d\nEmbedding size: %d\n' \
          'Batch size: %d\nNegative Sampling size: %d\n'\
          'Epochs to train: %d\n' \
          'Adam Optimizer epsilon: %f\n' \
          'Model save every: %d steps\n' \
          'Report progress every: %d steps' %\
          (FLAGS.model, FLAGS.context_size, FLAGS.emb_dim, FLAGS.batch_size,
           FLAGS.neg_size, FLAGS.epoch, FLAGS.eps, FLAGS.ck_point,
           FLAGS.report)

    # Generate the model and run the training
    word2vec = LexWord2Vec(model_folder_path,
                           word2idx,
                           context_size=FLAGS.context_size,
                           model=FLAGS.model,
                           emb_dim=FLAGS.emb_dim,
                           batch_size=FLAGS.batch_size,
                           neg_size=FLAGS.neg_size,
                           eps=FLAGS.eps,
                           rcm_ratio=FLAGS.rcm)

    # Start training
    embeds = word2vec.train(seq_data, pw_data, vocab_size, idx2word, count,
                            epoch=FLAGS.epoch, ck_point=FLAGS.ck_point,
                            report=FLAGS.report)

    # Display KNN for frequent words
    print '========Eval by KNN of Words========'
    word2vec.eval_knn(top_k=5)

    # Output two files: xxx.embeds binary / xxx.csv CSV
    embeds_file = os.path.join(FLAGS.output_dir, '%s.embeds' % FLAGS.task_name)
    csv_file = os.path.join(FLAGS.output_dir, '%s.csv' % FLAGS.task_name)
    dump(embeds, open(embeds_file, 'wb'))
    with open(csv_file, 'w') as output:
        for i in xrange(len(embeds)):
            output.write('%s\t%s\n' % (idx2word[i],
                                      '\t'.join(map(str, list(embeds[i])))))
    print '========Check Outputs========'
    print '%s' % embeds_file
    print '%s' % csv_file

    # Must end the session
    word2vec.end_session()

if __name__ == '__main__':
    main()
