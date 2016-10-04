# deeplearning-with-tensorflow
A deep learning tool built on top of Tensorflow, aiming at implementing various models and testing them on tasks including semantic matching and generative question answering.  

Implemented Models:  
* Word Embedding: SKIPGRAM, CBOW, RCM  
* TODO: Seq2Seq models  

## Content
<pre>
|-- <b>config</b> (YAML configurations for model parameters)
    |-- word_emb_def.yaml (an example of YAML configs) 
|-- <b>data</b> (default data input directory)
    |-- task_name (a folder of serialized data for a task)
|-- <b>embeds</b> (default word embedding output directory)
    |-- task_name.csv (embeddings per line with first column as the word itself)
    |-- task_name.embeds (a cPickle serialzed dictionary {word: embedding array})  
|-- <b>model</b> (default model saving directory)
    |-- task_name (a folder of saved models)
|-- <b>tf_nlp</b> (models implemented tensorflow )
    |-- proc_corpus (corpus pre-processing)
    |-- word_emb (models for training word embeddings)
    |-- cfg.py (global parameters)
    |-- utils.py (useful utils)    
|-- <b>third_party</b> (default directory of third party libs)
|-- <b>run_word_emb.py</b> (exec word embedding training)
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

## REFERENCE
Mikolov, T., and J. Dean. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems (2013).

Yu, Mo, and Mark Dredze. "Improving Lexical Embeddings with Semantic Knowledge." ACL (2). 2014.

## Contact
Yun Wang: yunwATemail.arizona.edu
