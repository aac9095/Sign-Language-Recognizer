import arpa
import os

def one_gram(df_prob):
    # TODO 1-gram lm
    lm = arpa.loadf(os.path.join("data", "ukn.1.lm"))
    print(lm[0].s("JOHN WRITE HOMEWORK"))
    print(lm[0].log_s("JOHN WRITE HOMEWORK"))

if __name__ == '__main__':
    one_gram(12)