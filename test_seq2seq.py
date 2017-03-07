#!/usr/bin/env python
"""Test a seq2seq model

Read model parameters from specified YAML, then run seq2seq test from shell.
"""
# Python default
import argparse
import os
import subprocess
# Third party
import yaml

__author__ = 'yunw@email.arizona.edu (Yun Wang)'

# Setup shell command parser
parser = argparse.ArgumentParser(description='Train seq2seq model')
parser.add_argument("-f", "--file",
                    dest="filename",
                    help="read configs from YAML",
                    default='./config/seq2seq_test_def.yaml',
                    metavar="FILE")
args = parser.parse_args()
file_name = args.filename

# Read YAML from specified location
if not file_name.endswith('.yaml') or not os.path.isfile(file_name):
    IOError('Can not find valid configuration files')

print 'Read configurations from %s' % file_name

with open(file_name, 'r') as yaml_file:
    yaml_config = yaml.safe_load(yaml_file)
    we_config = yaml_config['seq2seq']
    # Call the actual model to build word embeddings
    subprocess.call(['python', '-m', 'tf_nlp.seq2seq.run',
                     '--data_dir', '%s' % we_config['data_dir'],
                     '--model_dir', '%s' % we_config['model_dir'],
                     '--decode', '%d' % 1,
                     '--beam', '%d' % we_config['beam'],
                     '--test_file', '%s' % we_config['test_file'],
                     '--task_name', '%s' % we_config['task_name'],
                     '--third_paty', '%s' % we_config['third_paty'],
                     ])
