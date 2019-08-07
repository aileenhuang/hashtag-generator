# Eigen Coding Challenge
This is my submission for the Eigen Coding Challenge. Inside, you'll find the following contents:

1. `eigen-app`, a simple Flask app with a file uploader
2. `processor.py`, the backend module, which can also be called as a script

## `processor.py`
The processor incorporates two implementations of Latent Dirichlet Allocation, one from [Gensim](https://radimrehurek.com/gensim/) and the other from [Ariddell](https://github.com/lda-project/lda/tree/master).

Neither is perfect, though the Gensim implementation appears to be a bit more robust and produces topics for which words are more similarly related. However, the Flask application only calls Aridell's implementation because the output is simpler to work with. I would imagine that the differences between the two come from sampling methods and multithreading/async quirks.
