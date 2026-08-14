# Worked examples

The worked examples from the course notes at <https://nlp.jcrlabz.com>.

Each one reproduces a table printed in the book, so you can check any number the
text claims rather than take it on trust.

Every notebook is self contained. It writes its own source to disk and runs it,
so there is no data file to fetch and nothing to install beyond what the first
cell installs. Open one in Colab and choose **Runtime > Run all**.

Colab opens a notebook from GitHub read only. Click **Copy to Drive** to keep
your changes.

Read the chapter first. The notebook is the check, not the explanation.

## The examples

| notebook | what it shows | chapter |
| --- | --- | --- |
| [`bridging_the_gap`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/bridging_the_gap.ipynb) | One request, three answers | [1. What Natural Language Processing Is](https://nlp.jcrlabz.com/book/whatisnlp/) |
| [`edit_distance`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/edit_distance.ipynb) | Edit distance, every cell | [2. Corpora and Preprocessing](https://nlp.jcrlabz.com/book/corpora/) |
| [`empirical_laws`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/empirical_laws.ipynb) | Zipf and Heaps, fitted | [3. The Empirical Laws of Text](https://nlp.jcrlabz.com/book/empirical-laws/) |
| [`weighting`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/weighting.ipynb) | tf-idf and PMI, by hand | [4. Term Weighting and Similarity](https://nlp.jcrlabz.com/book/termweight/) |
| [`bpe`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/bpe.ipynb) | BPE, five merges | [5. Subword Tokenisation](https://nlp.jcrlabz.com/book/tokenization/) |
| [`hal`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/hal.ipynb) | HAL, ramped and asymmetric | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`coals`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/coals.ipynb) | COALS, all three steps | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`svd`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/svd.ipynb) | SVD, and what truncation costs | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`svd_rank`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/svd_rank.ipynb) | Choosing K from the spectrum | [7. Count Vectors, PPMI, and SVD](https://nlp.jcrlabz.com/book/countvectors/) |
| [`word2vec`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/word2vec.ipynb) | word2vec, one step at a time | [9. Learned Word Embeddings](https://nlp.jcrlabz.com/book/embeddings/) |
| [`factorisation`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/factorisation.ipynb) | Three methods, one matrix | [9. Learned Word Embeddings](https://nlp.jcrlabz.com/book/embeddings/) |
| [`ngram_lm`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/ngram_lm.ipynb) | Smoothing, perplexity and the U-curve | [10. n-gram Language Models and Perplexity](https://nlp.jcrlabz.com/book/ngram-lm/) |
| [`neural_lm`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/neural_lm.ipynb) | A neural LM, counted and trained | [11. Neural Language Models](https://nlp.jcrlabz.com/book/neural-lm/) |
| [`rnn`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/rnn.ipynb) | Unrolling, and the vanishing gradient | [12. Recurrent Networks](https://nlp.jcrlabz.com/book/recurrent/) |
| [`gated`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/gated.ipynb) | The gradient highway | [13. Gated Recurrence: LSTM and GRU](https://nlp.jcrlabz.com/book/gated/) |
| [`classification`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/classification.ipynb) | Accuracy, F1 and a confusion matrix | [14. Text Classification and Evaluation](https://nlp.jcrlabz.com/book/classification/) |
| [`contextual`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/contextual.ipynb) | What one vector per word costs | [15. Contextual Embeddings](https://nlp.jcrlabz.com/book/contextual/) |
| [`attention`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/attention.ipynb) | Self-attention, one query at a time | [16. Self-Attention and the Transformer](https://nlp.jcrlabz.com/book/self-attention/) |
| [`decoding`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/decoding.ipynb) | Greedy, beam, temperature, top-p | [18. Steering LLMs: Decoding and Prompting](https://nlp.jcrlabz.com/book/decoding-prompting/) |
| [`bleu`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/bleu.ipynb) | BLEU, ROUGE and their blind spots | [21. Evaluating Generated Text](https://nlp.jcrlabz.com/book/evaluation/) |
| [`ibm_model1`](https://colab.research.google.com/github/Ramaseshanr/anlp/blob/master/worked-examples/ibm_model1.ipynb) | EM learning an alignment | [24. Machine Translation](https://nlp.jcrlabz.com/book/translation/) |

## Where these come from

They are generated from the book source and pushed here on every publish, so
anything edited in this folder is overwritten. Report a problem against the
book at <https://nlp.jcrlabz.com>.
