#!/usr/bin/env python
"""Run a Word Embedding Training Task

Read model parameters from specified YAML, then run a word emb job from shell.
"""
# Python default
import argparse
import os
import subprocess
# Third party
import yaml

__author__ = 'yunwang@us.ibm.com (Yun Wang)'

# Setup shell command parser
parser = argparse.ArgumentParser(description='Train word embeddings')
parser.add_argument("-f", "--file",
                    dest="filename",
                    help="read configs from YAML",
                    default='./config/word_emb_def.yaml',
                    metavar="FILE")
args = parser.parse_args()
file_name = args.filename

# Read YAML from specified location
if not file_name.endswith('.yaml') or not os.path.isfile(file_name):
    IOError('Can not find valid configuration files')

print 'Read configurations from %s' % file_name

with open(file_name, 'r') as yaml_file:
    yaml_config = yaml.safe_load(yaml_file)
    we_config = yaml_config['word_emb']
    # Call the actual model to build word embeddings
    subprocess.call(['python', '-m', 'tf_nlp.word_emb.run',
                     '--batch_size', '%d' % we_config['batch_size'],
                     '--ck_point', '%d' % we_config['ck_point'],
                     '--context_size', '%d' % we_config['context_size'],
                     '--emb_dim', '%d' % we_config['emb_dim'],
                     '--eps', '%f' % float(we_config['eps']),
                     '--epoch', '%d' % we_config['epoch'],
                     '--input_dir', '%s' % we_config['input_dir'],
                     '--model_dir', '%s' % we_config['model_dir'],
                     '--model', '%s' % we_config['model'],
                     '--neg_size', '%d' % we_config['neg_size'],
                     '--output_dir', '%s' % we_config['output_dir'],
                     '--preload', '%d' % we_config['preload'],
                     '--report', '%d' % we_config['report'],
                     '--rcm', '%d' % we_config['rcm'],
                     '--task_name', '%s' % we_config['task_name'],
                     '--third_paty', '%s' % we_config['third_paty'],
                     '--vocab_size', '%d' % we_config['vocab_size']
                     ])
