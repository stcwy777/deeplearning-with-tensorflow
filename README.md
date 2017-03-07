# deeplearning-with-tensorflow
A deep learning tool built on top of Tensorflow, aiming at implementing various models and testing them on tasks including semantic matching and generative question answering.  

Implemented Models:  
* Word Embedding: SKIPGRAM, CBOW, RCM  
* Seq2Seq MT model 

## Content
<pre>
|-- <b>config</b> (YAML configurations for model parameters)
    |-- word_emb_def.yaml (example YAML configs for word embedding) 
    |-- seq2seq_train_def.yaml (example YAML configs for seq2seq training)
    |-- seq2seq_test_def.yaml (example YAML configs for seq2seq testing)        
|-- <b>data</b> (default data input directory)
    |-- task_name (a folder of serialized data for a task)
|-- <b>qa-data</b> (default data input directory for seq2seq)
|-- <b>embeds</b> (default word embedding output directory)
    |-- task_name.csv (embeddings per line with first column as the word itself)
    |-- task_name.embeds (a cPickle serialzed dictionary {word: embedding array})  
|-- <b>model</b> (default model saving directory)
    |-- task_name (a folder of saved models)
|-- <b>qa-model</b> (default model save directory for seq2seq)    
|-- <b>tf_nlp</b> (models implemented tensorflow )
    |-- proc_corpus (corpus pre-processing)
    |-- word_emb (models for training word embeddings)
    |-- seq2seq (models for seq2seq learning)
    |-- cfg.py (global parameters)
    |-- utils.py (useful utils)    
|-- <b>third_party</b> (default directory of third party libs)
|-- <b>run_word_emb.py</b> (exec word embedding training)
|-- <b>train_seq2seq.py</b> (exec seq2seq training)
|-- <b>test_seq2seq.py</b> (exec seq2seq testing)
</pre>

## Dependencies
+ Python packages
  - networkx, nltk, numpy, scikit-learn, tensorflow, pyyaml, rdflib
+ Stanford NLP tool (included in folder third_party)
  - stanford-ner, stanford-postagger
  
## Usage (list of currently supported tasks)
### Train Word Embedding
###### One step script (read default parameters from config/word_emb_def.yaml)  
```
./run_word_emb.py [-h --help HELP] [-f --file YAML FILE TO LOAD]
```
###### Model parameters  
```
python -m tf_nlp.word_emb.run --help
```
Note: Current implementation read all '.txt' files under data input directory as input corpus and all '.csv' files as input pairwise data for RCM models.  

### Seq2Seq Experiments (Train / Test)
###### One step script (read default parameters from config/seq2seq_train_def.yaml / seq2seq_test.def.yaml)  
###### Train a Seq2Seq model
```
./train_seq2seq.py [-h --help HELP] [-f --file YAML FILE TO LOAD]
```
###### Test a Seq2Seq model
```
./test_seq2seq.py [-h --help HELP] [-f --file YAML FILE TO LOAD]
```
###### Model parameters  
```
python -m tf_nlp.seq2seq.run --help
```
Some Notes:
1. tf_seq2seq_lib.py and tf_seq2seq_model.py are modified from TensorFLow seq2seq library for beam search support
2. beam search implementation credits to https://github.com/pbhatia243
3. Training will keep running until manually terminate it. Set up a proper check point (default 5000 training steps) to save you models
4. For model testing, depends on the user specified 'task_name' (default: test), a 'task_name-predict.txt' will be generated under the 'data_dir', each line is generated response. Another file 'task_name-filter.txt', contains the questioin and line number in original input. To compare the results, if you have ground truth about the responses, use 'task_name-filter' to match the right resposne in your ground truth and compare it with corresponding response in 'task_name-predic.txt'

## REFERENCE
Mikolov, T., and J. Dean. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems (2013).

Yu, Mo, and Mark Dredze. "Improving Lexical Embeddings with Semantic Knowledge." ACL (2). 2014.

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).


## Contact
Yun Wang: yunwATemail.arizona.edu
