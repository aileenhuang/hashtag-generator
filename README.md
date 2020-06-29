# Hashtags: A hashtag generator that uses Latent Dirichlet Allocation topic modeling
The goal of this project was to extract keywords and generate hashtags from a corpus of documents. Inside, you'll find the following contents:

1. `hashtags-app`, a simple Flask app with a file uploader
2. `hashtags-app/processors/topicgenerator.py`, the backend module that does all the text processing, which can also be called as a script

## `topicgenerator.py`
The `TopicGenerator` incorporates two implementations of Latent Dirichlet Allocation, one from [Gensim](https://radimrehurek.com/gensim/) and the other from [Ariddell](https://github.com/lda-project/lda/tree/master).

Neither is perfect, though the Gensim implementation appears to be a bit more robust and produces topics for which words are more similarly related. However, the Flask application only calls Aridell's implementation because the output is simpler to work with. I would imagine that the differences between the two come from sampling methods and multithreading/async quirks.

## Usage
To run, initialize virtual environment, which will install necessary dependencies.

```shell
$ ./init_env.sh
```

After this, to go into the development environment, just run
```shell
$ source dev3.7.env
```

To launch the app, call `flask run` and navigate to `127.0.0.1:5000` in your preferred browser.

You can also call the topic generator directly from within the `processors` directory and play around with the output.

## Output
My output is slightly modified from what was described in the coding challenge. Instead of prioritizing words, I prioritize documents and corresponding top topics, and then retrieve lists of top words/sentences based on those top topics.

## Things to improve
### Backend Enhancements
* Robust handling of invalid/corrupt files, redundant file names, generally not sane inputs
* Multilingual support
* Phrase-based LDA implementation. Currently working only with unigrams, which is really not so useful. The bag-of-words assumption that LDA rests on is quite limited, unfortunately
* Unit and integration tests
* Subclassing different topic models instead of wrapping everything under one class. Ideally, I should've created an abstract metaclass called `TopicModel` and made the functions under `TopicGenerator` class methods, subclassed under `TopicModel`. (Under this paradigm, I also should name `TopicGenerator` something like `LDATopicModel` as well.)

### Frontend Problems (because I'm not a web developer)
* Cleaning up the Flask app frontend itself so that it's not hideous
* Getting style.css to work
* Fixing egregious "None None None..." issue displaying on the uploaded.html page
* On uploads page, dynamically generate new upload buttons instead of having a fixed number of them with some JS
