# emoji2vec with PyTorch
Fork of the PyTorch implementation of 
## Note
This repo constitutes of an update from the work of Piotr Wiercinski, many thanks to him for his work. The original repo can be found [here](https://github.com/pwiercinski/emoji2vec_pytorch).
[emoji2vec: Learning Emoji Representations from their Description](https://arxiv.org/pdf/1609.08359.pdf) by Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and Sebastian Riedel. Please check out the original tensorflow implementation [here](https://github.com/uclmr/emoji2vec).

Updated to work with the latest version of gensim (4.0.0) and some other little deprecated functions.
added the TSN-E algorithm to visualize the emoji vectors in 2D space.
Currently working on implementing the 3D TSN-E algorithm to visualize the emoji vectors in 3D space.

## Pre-trained model

If you are interested in using the emoji vectors used in the paper,
they can be found in Gensim text/binary format in `./pre-trained/`. The
pre-trained vectors are meant to be used in conjunction with `word2vec`,
and are therefore 300-dimensional. Other dimensions can be trained 
manually, as explained below. These vectors correspond with the following 
hyperparameters:

```
params = {
    "out_dim": 300,
    "pos_ex": 4,
    "max_epochs": 40,
    "ratio": 1,
    "dropout": 0.0,
    "learning": 0.001
}
```

### Basic Usage
Once you've downloaded the pre-trained model, you can easily integrate 
emoji embeddings into your projects like so:

```
import gensim.models as gsm

e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
happy_vector = e2v['😂']    # Produces an embedding vector of length 300
```

## Prerequisites
There are several prerequisites to using the code:

- You must supply your own pretrained word vectors that are compatible
with the Gensim tool. For instance, you can download the Google News
word2vec dataset [here](https://code.google.com/archive/p/word2vec/). 
This must be in the binary format, rather than the .txt format.
- To download tweets using Tweepy, you must create a Twitter application
at [https://apps.twitter.com/](https://apps.twitter.com/), and place
the four generated keys in `secret.txt` in the directory where you 
run the Python script. However, you may not have to download the tweets,
since they are stored raw in a `pickle` file in the repository.

## CLI Arguments
Much of this code shares a common command line interface, which allows
you to supply hyperparameters for training and model 
generation/retrieval as well as file locations. The following can be
supplied:

- `-d`: directory for training data (default is `./data/training`)
- `-w`: path to the word embeddings (i.e. Google News word2vec)
- `-m`: file where we store mapping between index and emoji, for 
convenient caching between runs
- `-em`: file where we cache the vectorized phrases so we don't have to 
recompute each time, only change when you change the train, test, and 
dev files
- `-k`: output dimension of the emoji vectors we are training
- `-b`: number of positive examples in a training batch
- `-e`: number of training epochs
- `-r`: ratio between positive and negative training examples in a batch
- `-l`: learning rate
- `-dr`: dropout rate
- `-t`: threshold for classification, used in accuracy calculations
- `-ds`: name of the dataset we are training on, mainly for output 
folder

These are defined in `parameter_parser.py`.

## Model

The Emoji2Vec model, as well as a class for passing in hyperparameters,
can be found in `model.py`. The Emoji2Vec class is a PyTorch
implementation of our model. 

Important to note is that one can evaluate
the correlation between a phrase and an emoji in two ways: one can 
either input a raw vector and an emoji index (for general queries), 
or the index of a training phrase and the index of an emoji (indices
being the indices in the Knowledge Base). Typically, unless you are 
training the model on a totally different set of training examples, 
you'll want to use set `use_embeddings` to `False` in the constructor 
of the model. Otherwise, you'll have to pass in embeddings generated 
by the `generate_embeddings` function in `utils.py`. 


## Phrase2Vec

The `Phrase2Vec` class is a convenience wrapper to compute vector sums 
for phrases. The class can be constructed with two different vector
sets simultaneously: a word2vec Gensim object and an emoji vector Gensim
object. Alternatively, you can provide two filenames to do so. Query
like so:

```
vec = phrase2Vec['I am really happy right now! 😄']
```

## Train

To train a single model, run `train.py` with any combination of the 
hyperparameters above. For instance,

```
python3 train.py -k=300 -b=4 -r=1 -l=0.001 -ds=unicode -d=./data/training -t=0.5
```

will generate emoji vectors with dimension 300, and will train in 
batches of 8 (4 positive, 4 negative examples) at a learning rate of 
0.001. `./data/training/` must contain `train.txt`, `dev.txt`, and 
`test.txt`, the format of each being a tab-delimited, newline-delimited:

```
beating heart	🍮	False
```

The program will output various metrics, including accuracy (at the 
threshold provided), f1 score, and auc for a ROC curve. Additionally,
the program will generate a Gensim representation of the model, a
PyTorch representation of the model and a cache of the results of the model's predictions on the 
train and dev datasets.

These results can be found in the following folder:

```
./results/unicode/k-300_pos-4_rat-1_ep-40_dr=0/
```

## Grid Search

You can perform a grid search on a hyperparameter space one of two ways:
either directly modify the `search_params` variable in `grid_search.py`
and running `grid_search.py`, or from a separate file call `grid_search`
with supplied parameter set. In essence, this grid search will generate
results and embeddings in the same way as `train.py` for each parameter
combination. The searchable parameters are represented as follows:

```
search_params = {
    "out_dim": [300],
    "pos_ex": [4, 16, 64],
    "max_epochs": [10, 20],
    "ratio": [0, 1, 2],
    "dropout": [0.0, 0.1]
}
```

## Twitter Sentiment Dataset
`twitter_sentiment_dataset.py` contains a collection of helper functions
for downloading, processing, and reasoning about tweets. In general,
since tweets have already been downloaded and parsed and cached in
`./data/tweets/examples.p`, a client shouldn't need to access these
functions unless they are running them on a new set of Tweets


## Utils

`utils.py` contains several utility functions used in various files,
and generally need not be used externally.

## Contact
Contact me at `p.wierc \[at\] gmail.com` with questions about this implementation.
