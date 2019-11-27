# COMS 4995 Team 4 Deep Learning Project
### Topic Level Sentiment Analysis Using Zero Shot Classifier
### Kevin Wong, Jiali Sun, Larissa

##### Kevin Wong, Scarlett Sun and Larissa Liu contributed equally to this project.

Original Paper link: [arXiv:1903.12626](https://arxiv.org/abs/1903.12626)

This Repository is forked from (https://github.com/JingqingZ/KG4ZeroShotText)

There is another Repository for the second stage of Sentiment Analysis. (https://github.com/kevinwong2013/COMS4995-Deep-Learning-Team-4.git)
## Contents
1. [Abstract](#Abstract)
2. [Code](#Code)
3. [Acknowledgement](#Acknowledgement)
4. [Citation](#Citation)

<h2 id="Abstract">Abstract</h2>

State of the art sentiment analysis models
achieved more than 90% accuracy for some benchmark dataset. However, it would be much more
informative if finer grained sentiment analysis on
aspect level can be done. This text  classification problem is frequently
challenged by the need to determine whether the document is related to the
queried aspect. The challenge is further complicated by insufficient or even unavailable training 
data of the queried aspect.

Recognising text documents of classes that have 
never been seen in the learning stage, so-called zero-shot text 
classification, is therefore difficult and only limited previous 
works tackled this problem. This report investigates using zero
shot learning for the detection of sentences directly related to the specific aspects and perform
aspect-level sentiment analysis.

<h2 id="Code">Code</h2>

### Checklist

In order to run the code, please check the following issues.

- [x] Package dependencies:
    - Python 3.5
    - TensorFlow 1.11.0
    - [TensorLayer] 1.11.0
    - Numpy 1.14.5
    - Pandas 0.21.0
    - NLTK 3.2.5
    - tqdm 2.2.3
    - [gensim](https://pypi.org/project/gensim/) 3.7.1
    - [language-check](https://pypi.org/project/language-check/) 1.1
- [x] Download original datasets
    - [GloVe.6B.200d](https://nlp.stanford.edu/projects/glove/)
    - [ConceptNet v5.6.0](https://github.com/commonsense/conceptnet5/wiki/Downloads)
    - [DBpedia ontology dataset](https://github.com/zhangxiangxiao/Crepe)
    - [20 Newsgroups original 19997 docs](http://qwone.com/~jason/20Newsgroups/)
- [x] Check [config.py] and update the locations of data files accordingly. The [config.py] also defines the locations of intermediate files.
- [x] All files together are zipped and have been uploaded to [supplementary].
- [x] Other intermediate files should be generated automatically when they are needed.

Please feel free to raise an issue if you find any difficulty to run the code or get the intermediate files.

[supplementary]: https://drive.google.com/open?id=114B4oocAdqlwLdzVcHG5m2HDg_zRBvm1
[TensorLayer]: https://github.com/tensorlayer/tensorlayer
[config.py]: src_reject/config.py
[playground.py]: src_reject/playground.py

### How to perform data augmentation

An example:
```bash
python3 topic_translation.py \
        --data dbpedia \
        --nott 100
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `nott`: No. of original texts to be translated into all classes except the original class. If `nott` is not given, all the texts in the training dataset will be translated. 

The location of the result file is specified by config.\{zhang15_dbpedia, news20\}_train_augmented_aggregated_path.

Three outputs files will be automatically generated (filepath defined in [config.py](src_reject/config.py)).
* config.word_embed_gensim_file_path
* config.POS_OF_WORD_path
* config.WORD_TOPIC_TRANSLATION_path


### How to perform feature augmentation / create v_{w,c}

An example:
```bash
python3 kg_vector_generation.py --data imdb 
```
The argument of the command represents
* `data`: Dataset, either `dbpedia` or `20news`.

The locations of the result files are specified by config.\{imdb\}_kg_vector_dir.

### How to train / test Phase 1

- Without data augmentation: an example
```bash
python3 train_reject.py \
        --data imdb \
        --unseen 0.5 \
        --model vw \
        --nepoch 3 \
        --rgidx 1 \
        --train 1
```

- With data augmentation: an example
```bash
python3 train_reject_augmented.py \
        --data imdb \
        --unseen 0.5 \
        --model vw \
        --nepoch 3 \
        --rgidx 1 \
        --naug 100 \
        --train 1
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either `0.25` or `0.5`.
* `model`: The model to be trained. This argument can only be
    * `vw`: the inputs are embedding of words (from text)
* `nepoch`: The number of epochs for training
* `train`: In Phase 1, this argument does not affect the program. The program will run training and testing together.
* `rgidx`: Optional, Random group starting index: e.g. if 5, the training will start from the 5th random group, by default `1`. This argument is used when the program is accidentally interrupted.
* `naug`: The number of augmented data per unseen class

The location of the result file (pickle) is specified by config.rejector_file. The pickle file is actually a list of 10 sublists (corresponding to 10 iterations). Each sublist contains predictions of each test case (1 = predicted as seen, 0 = predicted as unseen).

### How to train / test the zero-shot classifier 

An example:
```bash
python3 train_unseen.py \
        --data imdb \
        --unseen 0.5 \
        --model vwvcvkg \
        --ns 2 --ni 2 --sepoch 10 \
        --rgidx 1 --train 1
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either `0.25` or `0.5`.
* `model`: The model to be trained. This argument can be (correspond with Table 6 in the paper)
    * `kgonly`: the inputs are the relationship vectors which are extracted from knowledge graph (KG).
    * `vcvkg`: the inputs contain the embedding of class labels and the relationship vectors.
    * `vwvkg`: the inputs contain the embedding of words (from text) and the relationship vectors.
    * `vwvc`: the inputs contain the embedding of words and class labels.
    * `vwvcvkg`: all three kinds of inputs mentioned above.
* `train`: 1 for training, 0 for testing.
* `sepoch`: Repeat training of each epoch for several times. The ratio of positive/negative samples and learning rate will keep consistent in one epoch no matter how many times the epoch is repeated.
* `ns`: Optional, Integer, the ratio of positive and negative samples, the higher the more negative samples, by default `2`. 
* `ni`: Optional, Integer, the speed of increasing negative samples during training per epoch, by default `2`.
* `rgidx`: Optional, Random group starting index: e.g. if 5, the training will start from the 5th random group, by default `1`. This argument is used when the program is accidentally interrupted.
* `gpu`: Optional, GPU occupation percentage, by default `1.0`, which means full occupation of available GPUs.
* `baseepoch`: Optional, you may want to specify which epoch to test.