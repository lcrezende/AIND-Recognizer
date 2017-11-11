import warnings
from asl_data import SinglesData
from hmmlearn.hmm import GaussianHMM

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    items = test_set.get_all_Xlengths()
    for item in items:
        best_score = float('-inf')
        best_guess = None
        item_prob = {}
        item_X, item_lengths = items[item]
        for word, model in models.items():
            try:
                score = model.score(item_X)
            except:
                score = float('-inf')
            item_prob[word] = score
            if score > best_score:
                best_score = score
                best_guess = word
        probabilities.append(item_prob)
        guesses.append(best_guess)

    return probabilities, guesses
    # raise NotImplementedError
