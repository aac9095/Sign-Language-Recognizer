import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None 


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bestBIC, bestComponent = float('inf'), self.n_constant

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            bic = self.bic_model(n_component)
            if bic < bestBIC:
                bestBIC = bic
                bestComponent = n_component

        return self.base_model(bestComponent)

    def bic_model(self, n_component):
        try:
            model = GaussianHMM(n_components=n_component, n_iter=1000).fit(self.X, self.lengths)
            logL = model.score(self.X, self.lengths)
            n_parameters = n_component * (n_component + 2 * len(self.X[0])) - 1
            bic = n_parameters * math.log(len(self.X)) - 2 * logL
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_component))
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_component))
            bic = float('inf')

        return bic


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        bestDIC, bestComponent = float('-inf'), self.n_constant

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            dic = self.dic_model(n_component)
            if dic > bestDIC:
                bestDIC = dic
                bestComponent = n_component

        return self.base_model(bestComponent)

    def dic_model(self,n_component):
        """
        :param n_component:
        :return: dic score of the model
        """
        try:
            model = GaussianHMM(n_components=n_component, n_iter=1000).fit(self.X, self.lengths)
            logL = model.score(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_component))
            avg_score = self.avg_score_for_other_words(model)
            dic = logL - avg_score
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_component))
            dic = float('-inf')

        return dic

    def avg_score_for_other_words(self,model):
        """
        :param model:
        :return: avg score for other words
        """

        logL_others = []

        for word in self.hwords:
            if word is not self.this_word:
                X_other, length_other = self.hwords[word]
                logL_others.append(model.score(X_other, length_other))

        return np.mean(logL_others)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        maxLogL, bestComponent = float('-inf'), self.n_constant
        n_splits = min(3, len(self.sequences))

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            avg_logL = self.cross_validation(n_component, n_splits)
            if avg_logL > maxLogL:
                bestComponent = n_component
                maxLogL = avg_logL

        return self.base_model(bestComponent)

    def cross_validation(self, n_component, n_splits):
        temp_logL, kfold = [], KFold(n_splits=n_splits)
        try:
            for trainIndexes, testIndexes in kfold.split(self.sequences):
                temp_logL.append(self.cv_model(trainIndexes, testIndexes, n_component))
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_component))
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_component))
            return float('-inf')

        return np.mean(temp_logL)

    def cv_model(self, trainIndexes, testIndexes, n_component):
        X_train, lengths_train = combine_sequences(trainIndexes, self.sequences)
        X_test, lengths_test = combine_sequences(testIndexes, self.sequences)
        model = GaussianHMM(n_components=n_component, n_iter=1000).fit(X_train, lengths_train)
        logL = model.score(X_test, lengths_test)

        return logL
