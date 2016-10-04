"""Package Level Parameters

This module provides definition of configurations (global variable) to use
package-wise.
"""

# TODO: use YAML to load configs

__author__ = 'yunwang@us.ibm.com (Yun Wang)'

DATA_DIR = './data'                     # Default data directory
MODEL_NAME = 'model'                    # Default model directory
KB_LABEL = 'rdf-schema#label'           # Label string defined in KB

W_UNK = '_unk_'                         # Makeup word for out bound vocabulary
W_EOS = '_eos_'                         # Makeup word for end of line
W_EOT = '_eot_'                         # Makeup word for end of turn

DEF_VOCAB_SIZE = 50000                  # Default vocabulary size

# Directories of standford-libs
THIRD_PATY_PATH = './third_party'
NER_JAR = 'stanford-ner-3.6.0.jar'
NER_MODEL = 'english.all.3class.distsim.crf.ser.gz'

POS_JAR = 'stanford-postagger-3.6.0.jar'
POS_MODEL = 'english-bidirectional-distsim.tagger'

# Word embeddings parameters
ONLY_RCM = -1
NO_RCM = 0
