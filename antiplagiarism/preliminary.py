import pandas as pd
import string
import numpy as np


def remove_punctuation(s: str) -> str:
    """
    IN: a string s
    OUT: the string s without punctuation
    """
    return s.translate(
        str.maketrans('', '', string.punctuation)
        )


def main():
    with open('commedia.txt', 'r') as f:
        lines = f.readlines()
    df = pd.Series(lines)
    df = df.apply(remove_punctuation) \
                 .apply(lambda str: str[:-1])
    n_verses = len(df)
    print('Number of verses: ', n_verses)
    df_split: pd.Series = df.apply(lambda str: str.split()) \
                 .explode(ignore_index=True) \
                 .dropna()
    distinct_words = np.unique(df_split.values)
    print('Number of words: ', len(df_split))
    print('Number of distinct words: ', len(distinct_words))
    i = 0
    sentences = []
    WINDOW_SIZE = 4
    STRIDE = 1
    while i < len(distinct_words) - WINDOW_SIZE:
        j = i
        sentence = []
        for j in range(WINDOW_SIZE):
            sentence.append(distinct_words[i+j])
        sentences.append(sentence)
        i = i + STRIDE
    

if __name__ == '__main__':
    main()
