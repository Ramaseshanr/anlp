# Applied Natural Language Processing

Code for the course notes *Natural Language Processing: From Corpus Statistics
to Grounded LLMs*, which you can read in full at <https://nlp.jcrlabz.com>.

## Read the book

Online at <https://nlp.jcrlabz.com>, or as a PDF:
**[download the latest build](https://github.com/Ramaseshanr/anlp/releases/download/book-latest/nlp-course-notes.pdf)**.
The PDF is republished from the same source every time the site is, so the two
never disagree.

## What is here

`book/worked-examples/` holds one notebook per worked example. Each reproduces a table
printed in the book, so you can check any number the text claims rather than
take it on trust.

`Archive/` holds the demonstration notebooks from earlier runs of the course.
They are kept for reference and are not maintained.

## Running an example

Click any notebook below and it opens in Colab. Choose **Runtime > Run all**.

Every notebook is self contained. The first cell defines everything the example
needs and runs in the notebook itself, so there is no data file to fetch and
nothing to install beyond what the install cell installs.

Colab opens a notebook from GitHub read only. Click **Copy to Drive** to keep
your changes.

Read the chapter first. The notebook is the check, not the explanation.

## The worked examples

| notebook | what it shows | chapter |
| --- | --- | --- |
| [`bridging_the_gap`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/bridging_the_gap.ipynb) | One request, three answers | [1. What Natural Language Processing Is](https://nlp.jcrlabz.com/book/whatisnlp/) |
| [`edit_distance`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/edit_distance.ipynb) | Edit distance, every cell | [2. Corpora and Preprocessing](https://nlp.jcrlabz.com/book/corpora/) |
| [`empirical_laws`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/empirical_laws.ipynb) | Zipf and Heaps, fitted | [3. The Empirical Laws of Text](https://nlp.jcrlabz.com/book/empirical-laws/) |
| [`weighting`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/weighting.ipynb) | tf-idf and PMI, by hand | [4. Term Weighting and Similarity](https://nlp.jcrlabz.com/book/termweight/) |
| [`bpe`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/bpe.ipynb) | BPE, five merges | [5. Subword Tokenisation](https://nlp.jcrlabz.com/book/tokenization/) |
| [`hal`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/hal.ipynb) | HAL, ramped and asymmetric | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`coals`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/coals.ipynb) | COALS, all three steps | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`svd`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/svd.ipynb) | SVD, and what truncation costs | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`svd_rank`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/svd_rank.ipynb) | Choosing K from the spectrum | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`word2vec`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/word2vec.ipynb) | word2vec, one step at a time | [9. Learned Word Embeddings](https://nlp.jcrlabz.com/book/embeddings/) |
| [`factorisation`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/factorisation.ipynb) | Three methods, one matrix | [9. Learned Word Embeddings](https://nlp.jcrlabz.com/book/embeddings/) |
| [`ngram_lm`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/ngram_lm.ipynb) | Smoothing, perplexity and the U-curve | [10. n-gram Language Models and Perplexity](https://nlp.jcrlabz.com/book/ngram-lm/) |
| [`neural_lm`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/neural_lm.ipynb) | A neural LM, counted and trained | [11. Neural Language Models](https://nlp.jcrlabz.com/book/neural-lm/) |
| [`rnn`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/rnn.ipynb) | Unrolling, and the vanishing gradient | [12. Recurrent Networks](https://nlp.jcrlabz.com/book/recurrent/) |
| [`gated`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/gated.ipynb) | The gradient highway | [13. Gated Recurrence: LSTM and GRU](https://nlp.jcrlabz.com/book/gated/) |
| [`classification`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/classification.ipynb) | Accuracy, F1 and a confusion matrix | [14. Text Classification and Evaluation](https://nlp.jcrlabz.com/book/classification/) |
| [`contextual`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/contextual.ipynb) | What one vector per word costs | [15. Contextual Embeddings](https://nlp.jcrlabz.com/book/contextual/) |
| [`attention`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/attention.ipynb) | Self-attention, one query at a time | [16. Self-Attention and the Transformer](https://nlp.jcrlabz.com/book/self-attention/) |
| [`decoding`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/decoding.ipynb) | Greedy, beam, temperature, top-p | [18. Steering LLMs: Decoding and Prompting](https://nlp.jcrlabz.com/book/decoding-prompting/) |
| [`bleu`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/bleu.ipynb) | BLEU, ROUGE and their blind spots | [21. Evaluating Generated Text](https://nlp.jcrlabz.com/book/evaluation/) |
| [`ibm_model1`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/book/worked-examples/ibm_model1.ipynb) | EM learning an alignment | [24. Machine Translation](https://nlp.jcrlabz.com/book/translation/) |

## About this file

This README and the notebooks in `book/worked-examples/` are generated from the book
source and pushed here on every publish, so edits made to them here are
overwritten. Everything else in the repository is edited by hand and is left
alone. Report a problem with an example against the book at <https://nlp.jcrlabz.com>.
