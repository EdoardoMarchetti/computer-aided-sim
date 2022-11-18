import pandas as pd
from os.path import sep


class BirthdayDistribution:
    __instance = None
    """
    Singleton class for handling a custom distribution file.
    It is simply a binding to a data source,
    it is useful to access the file only once.
    """
    def __init__(self):
        if BirthdayDistribution.__instance is not None:
            raise Exception('This is a singleton class.\
                Use BirthdayDistribution.get() instead.')
        df = pd.read_csv(
            f'data{sep}estimated_realistic_distribution.csv')
        self.alphabet = df['day_number'].values
        self.probabilities = df['probability'].values

    @staticmethod
    def get():
        if BirthdayDistribution.__instance is None:
            BirthdayDistribution.__instance = BirthdayDistribution()
        return BirthdayDistribution.__instance
