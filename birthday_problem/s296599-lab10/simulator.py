import numpy as np
from birthday_distribution import BirthdayDistribution
from tqdm import tqdm


class ConflictSimulator:

    class UnhandledDistributionException(Exception):
        pass

    def __init__(
            self,
            m: int,                    # the (fixed) number of people
            k: int,                    # the number of experiments
            seed: int,                 # seed for the random generator
            distribution: str,         # a str in ['uniform', 'realistic']
            verbose: bool):
        """
        Setup the simulation parameters, which are:
        - m: the number of samples in the population (required only
             for running the second experiment)
        - k: the number of times to repeat the experiment
        - distribution: 'uniform' for using the uniform distribution,
                        'realistic' for using the estimated real distribution
                        (requires the distribution file to be in the data folder)
        - seed: the seed for the random generator
        - verbose: set to true if you want a progress bar during simulation
        """
        self.m = m
        self.k = k
        self.verbose = verbose
        self.generator = np.random.default_rng(seed=seed)
        if distribution == 'uniform':
            self.distribution = lambda size: \
                self.generator.integers(
                    low=1,
                    high=365,
                    endpoint=True,
                    size=size
                    )
        elif distribution == 'realistic':
            realistic_distribution = BirthdayDistribution.get()
            self.distribution = lambda size: \
                self.generator.choice(
                    a=realistic_distribution.alphabet,
                    p=realistic_distribution.probabilities,
                    size=size
                    )
        else:
            raise self.UnhandledDistributionException\
                (f'{distribution} distribution is not\
                    handled by this class.')

    def exec_sim_1(self) -> float:
        """
        Compute the average number of people needed to produce
        the first birthday collision.
        It repeat the experiment k times and produce as output
        an array A of size k with all the results
        (i.e., A[i] is the number of people needed to produce
        the first collision in the experiment i in {1..k}).
        It returns the average of that array
        """
        self.sim_1_samples = np.empty(shape=(self.k,), dtype=int)
        for j in tqdm(range(self.k)) if self.verbose else range(self.k):
            conflict = False
            S = set()
            i = 0
            while not conflict:
                x = self.distribution(size=None)
                if x in S:
                    self.sim_1_samples[j] = i
                    conflict = True
                S.add(x)
                i += 1
        return np.mean(self.sim_1_samples)

    def exec_sim_2(self) -> float:
        """
        Compute the probability of a conflict in function of
        the population size [m].
        """
        accumulator = 0
        for _ in tqdm(range(self.k)) if self.verbose else range(self.k):
            L = self.distribution(size=(self.m, ))
            S = np.unique(L)
            accumulator += int(len(S) < len(L))
        return accumulator/self.k
