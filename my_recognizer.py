import warnings
from asl_data import SinglesData


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
    for word_id in range(len(test_set.wordlist)):
        X, lengths = test_set.get_item_Xlengths(word_id)
        bestScore, bestGuess, dic = float('-inf'), None, {}
        for key, model in models.items():
            try:
                dic[key] = model.score(X, lengths)
            except:
                dic[key] = float('-inf')
            if  bestScore < dic[key]:
                bestScore = dic[key]
                bestGuess = key
        guesses.append(bestGuess)
        probabilities.append(dic)
    # return probabilities, guesses
    return probabilities, guesses
