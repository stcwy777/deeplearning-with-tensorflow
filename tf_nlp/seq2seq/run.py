"""Run seq2seq experiments

This is the script for seq2seq experiments using TensorFlow seq2seq lib
It relies on Class Corpus to generated vocabulary, word indices and training
data from raw corpus and then uses Class LexWord2Vec to train word embeddings.

Users can specify the interval for saving trained model to a local directory.
By default, training status include average loss will be displayed every 500
training steps.

This module provides functions for internal uses (e.g. hard coded parameters):
1) process_corpus: check training data availability
2) read_data: read training data (integer index converted from words)
3) create_model: create new TensorFlow model or load from trained moodel
4) train: train the seq2seq model
5) decode: decode a input document by lines with / without beam search

Usage:
from the project root folder (.../cogn-dialog), type:
    python -m tf_nlp.seq2seq.run --help
to see how different parameters feed the script.

Decode output:
Two output files will be generated under the 'data_dir' (they are named by the
task_name parameter):
1) task_name-predict.txt: each line contains a predicted sentence
2) task_name-filter.txt: each line contains two columns: a source sentence
   (question) and the line number in original source file. I keep this
   information because the bucketing mechanism might skip input outside any
   predefined buckets (e.g. too short / long sentences).

"""

# Python default
import math
import os
import sys
import time

# Third party
from tensorflow.models.rnn.translate import data_utils

# Inner reference
from tf_nlp.proc_corpus.corpus import Corpus
from tf_nlp.seq2seq.tf_seq2seq_model import *
from tf_nlp.utils import _UNK, EOS_ID

# Use TF API to conveniently define parameters
tf.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                      "Learning rate decays by this much.")
tf.flags.DEFINE_float("max_gradient_norm", 5.0,
                      "Clip gradients to this norm.")
tf.flags.DEFINE_integer("batch_size", 80,
                        "Batch size to use during training.")
tf.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.flags.DEFINE_integer("que_vocab_size", 3000, "Query vocabulary size.")
tf.flags.DEFINE_integer("ans_vocab_size", 3000, "Answer vocabulary size.")
tf.flags.DEFINE_string("data_dir", "./qa-data", "Data directory")
tf.flags.DEFINE_string("model_dir", "./qa-model", "Model directory")
tf.flags.DEFINE_integer("max_train_data_size", 0,
                        "Limit on the size of training data (0: no limit).")
tf.flags.DEFINE_integer("ck_point", 5000,
                        "How many training steps to do per checkpoint.")
tf.flags.DEFINE_integer("decode", 0,
                        "Set to True for interactive decoding.")
tf.flags.DEFINE_integer("beam", 0,
                        "Set beam size (default 0)")
tf.flags.DEFINE_string("test_file", 'test_src.txt',
                       "Source file to decode (default 'test_src.txt')")
tf.flags.DEFINE_string("task_name", 'test',
                       "used to output results (default 'test_src.txt'")
tf.flags.DEFINE_string('third_party', os.path.abspath('./third_party'),
                       'third party libs (default: "./third_party")')

FLAGS = tf.flags.FLAGS

# Buckets used for padding sentences
_buckets = [(5, 15), (10, 25), (15, 35)]

# Maximum question: 15 words / maximum answers: 35 words
src_cut = 15
tar_cut = 35


def process_corpus(intput_dir, category, line_cut):
    """Check existence of training data (converted ids from words). If not, load
    raw text, generate vocabulary and convert raw text into ids.
    Args:
        input_dir: input data directory
        category: train / validation / test, just make sure it matches your file
        line_cut: maximum words for line(sentence)
    """
    corp = Corpus(FLAGS.third_party)
    # Regex rules that remove html tags
    corp.set_filt_rule('<.*?>')
    corp.set_filt_rule('\[|\]|\$')
    corp.set_filt_rule("``.*?''")

    print 'Preprocess %s......' % category

    # If input data not ready
    train_src = os.path.join(intput_dir, '%s.txt' % category)
    sents = corp.load_by_lines(train_src, use_filter=True, use_stem=False,
                               use_lemma=False, add_eos=False, add_eot=False,
                               conv_num=False, store_words=True, cutoff=line_cut)

    if not os.path.exists('%s-voc.txt' % category):
        print 'Generating vocabulary......'
        corp.build_vocab_to_file(os.path.join(intput_dir,
                                              '%s-voc.txt' % category))
    else:
        corp.load_vocab_from_file(os.path.join(intput_dir,
                                               '%s-voc.txt' % category))
    print 'Convert word into integers......'
    corp.doc_to_ids(sents, os.path.join(intput_dir, '%s-ids.txt' % category))


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path of the files with token-ids for the source language.
      target_path: path of the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()[:src_cut]]
                target_ids = [int(x) for x in target.split()[:tar_cut]]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(
                        _buckets):
                    if len(source_ids) < source_size and len(
                            target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only, beam_search, beam_size=10):
    """Create translation model and initialize or load parameters in session.

    Args:
      session: TensorFlow session
      forward_only: boolean option, True for decoding
      beam_search: boolean option, True for beam search
      beam_size: size of
    """

    model = Seq2SeqModel(
        FLAGS.que_vocab_size, FLAGS.ans_vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        forward_only=forward_only, beam_search=beam_search, beam_size=beam_size)
    print FLAGS.model_dir
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train(train_q, train_a, valid_q, valid_a):
    """Train a seq2seq model. source / target files contains tokens of ids

    Args:
        train_q: "questions" (source) for training
        train_a: "answers" (target) for training
        valid_q: "questions" (source) for validation
        valid_a: "answers" (target) for validation
    """
    with tf.Session() as sess:
        # Create model.
        print "Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size)
        # model = create_model(sess, False)
        model = create_model(sess, False, beam_search=False)
        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        dev_set = read_data(valid_q, valid_a)
        train_set = read_data(train_q, train_a, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that
        # we'll use to select a bucket. Length of [scale[i], scale[i+1]] is
        # proportional to the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) /
                               train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        print "Start training\n"
        while True:
            # Choose a bucket according to data distribution. We pick a random
            # number in [0, 1] and use the corresponding interval in
            # train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            # print "Get one batch data:\n", start_time
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False, False)
            print "Train one step:%d\n" % current_step
            step_time += (time.time() - start_time) / FLAGS.ck_point
            loss += step_loss / FLAGS.ck_point
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and evals.
            if current_step % FLAGS.ck_point == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print "global step %d learning rate %.4f step-time %.2f " \
                      "perplexity %.2f" % (model.global_step.eval(),
                                           model.learning_rate.eval(),
                                           step_time, perplexity)
                # Decrease learning rate if no improvement over last 3 times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.model_dir, "gen_qa.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on validation set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print "  eval: empty bucket %d" % bucket_id
                        continue
                    encoder_inputs, decoder_inputs, target_weights = \
                        model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs,
                                                 decoder_inputs,
                                                 target_weights, bucket_id,
                                                 True, False)
                    eval_ppx = math.exp(eval_loss) \
                        if eval_loss < 300 else float('inf')
                    print "eval: bucket %d perplexity %.2f" % \
                          (bucket_id, eval_ppx)
                sys.stdout.flush()


def decode(input_file, beam_size=0):
    """Decode input sentences to generate response

    Args:
      input_file: input file, one sentence per line
      output_file: file of generated responses
      filter_file: path of the
      beam_size: possible paths in beam search (0 indicates regular search)
    """

    # Load vocabularies from training data
    src_corp = Corpus(FLAGS.third_party)
    tar_corp = Corpus(FLAGS.third_party)
    _, src_vocab, src_rev_vocab = src_corp.load_vocab_from_file(
        os.path.join(FLAGS.data_dir, 'train-src-voc.txt'))

    _, tar_vocab, tar_rev_vocab = tar_corp.load_vocab_from_file(
        os.path.join(FLAGS.data_dir, 'train-tar-voc.txt'))

    with tf.Session() as sess:
        # Create model and load parameters.
        if beam_size > 0:
            model = create_model(sess, True, beam_search=True,
                                 beam_size=beam_size)
        else:
            model = create_model(sess, True, beam_search=False)
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        print FLAGS.data_dir, input_file
        input_path = os.path.join(FLAGS.data_dir, input_file)
        output_path = os.path.join(FLAGS.data_dir,
                                   '-'.join([FLAGS.task_name, 'predict.txt']))
        filter_path = os.path.join(FLAGS.data_dir,
                                   '-'.join([FLAGS.task_name, 'filter.txt']))

        sentences = []
        with open(input_path, 'r') as in_file:
            for line in in_file:
                sentences.append(line.rstrip())
        with open(output_path, 'w') as out_file, open(filter_path,
                                                      'w') as fil_file:
            seq = 0
            for sentence in sentences:
                # print sentence
                seq += 1
                # Get token-ids for the input sentence.
                conv_sent = []
                for word in sentence.split():
                    try:
                        conv_sent.append(src_vocab[word])
                    except KeyError:
                        conv_sent.append(src_vocab[_UNK])

                # Which bucket does it belong to?
                try:
                    bucket_id = min([b for b in xrange(len(_buckets))
                                     if _buckets[b][0] > len(conv_sent)])
                except ValueError:
                    continue

                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(conv_sent, [])]}, bucket_id)

                if beam_size == 0:
                    # Get output logits for the sentence.
                    _, _, output_logits = model.step(sess, encoder_inputs,
                                                     decoder_inputs,
                                                     target_weights, bucket_id,
                                                     True, False)
                    # A greedy decoder - argmax of output logits.
                    outputs = [int(np.argmax(logit, axis=1)) for logit in
                               output_logits]
                    # If there is an EOS symbol, cut them at that point.
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    # Print out French sentence corresponding to outputs.
                    predict = " ".join(
                        [tf.compat.as_str(tar_rev_vocab[output])
                         for output in outputs])
                else:
                    # Perform beam search
                    path, symbol, output_logits = model.step(sess,
                                                             encoder_inputs,
                                                             decoder_inputs,
                                                             target_weights,
                                                             bucket_id,
                                                             True, True)
                    # Extract symbols from beam
                    paths = []
                    for _ in xrange(beam_size):
                        paths.append([])
                    curr = range(beam_size)
                    num_steps = len(path)
                    for i in range(num_steps-1, -1, -1):
                        for j in range(beam_size):
                            paths[j].append(symbol[i][curr[j]])
                            curr[j] = path[i][curr[j]]
                    recos = set()

                    print "Unique results among beam size%d:" % beam_size
                    for i in range(beam_size):
                        foutputs = [int(logit) for logit in paths[i][::-1]]

                        # If there is an EOS symbol, cut them at that point.
                        if EOS_ID in foutputs:
                            foutputs = foutputs[:foutputs.index(EOS_ID)]
                        rec = " ".join([tf.compat.as_str(tar_rev_vocab[output])
                                        for output in foutputs])
                        if rec not in recos:
                            recos.add(rec)
                            print len(recos), rec

                    # TODO: No hint for the best path in beam
                    predict = list(recos)[0]
                print 'source: ', sentence
                print 'target:', predict

                # print("> ", end="")
                sys.stdout.flush()
                fil_file.write('%s %d\n' % (sentence, seq))
                out_file.write('%s\n' % predict)


def main():

    # Examine existence of key folders
    if not os.path.exists(FLAGS.data_dir):
        raise IOError('Input data directory (%s) not exists!' % FLAGS.data_dir)

    if not os.path.exists(FLAGS.model_dir):
        raise IOError('Model directory (%s) not exists!' % FLAGS.model_dir)

    # model_folder_path = os.path.join(FLAGS.model_dir, FLAGS.task_name)

    if FLAGS.decode != 0:
        print 'Start loading test input......'
        if not os.path.exists(os.path.join(FLAGS.data_dir,
                                           'train-tar-voc.txt')) or \
                not os.path.exists(os.path.join(FLAGS.data_dir,
                                                'train-src-voc.txt')):
            raise IOError('Vocabulary not found, train your model first')

        decode(FLAGS.test_file, FLAGS.beam)
    else:
        print 'Start processing training input......'
        # Check if we need to preprocess input
        if not os.path.exists(os.path.join(FLAGS.data_dir,'train-src-ids.txt')):
            process_corpus(FLAGS.data_dir, 'train-src', tar_cut)

        if not os.path.exists(os.path.join(FLAGS.data_dir,'train-tar-ids.txt')):
            process_corpus(FLAGS.data_dir, 'train-tar', tar_cut)

        if not os.path.exists(os.path.join(FLAGS.data_dir,'valid-src-ids.txt')):
            process_corpus(FLAGS.data_dir, 'valid-src', src_cut)

        if not os.path.exists(os.path.join(FLAGS.data_dir,'valid-tar-ids.txt')):
            process_corpus(FLAGS.data_dir, 'valid-tar', src_cut)

        # print src_size, tar_size
        train(os.path.join(FLAGS.data_dir, 'train-src-ids.txt'),
              os.path.join(FLAGS.data_dir, 'train-tar-ids.txt'),
              os.path.join(FLAGS.data_dir, 'valid-src-ids.txt'),
              os.path.join(FLAGS.data_dir, 'valid-tar-ids.txt'))

if __name__ == "__main__":
    main()
