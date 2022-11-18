from simulator import ConflictSimulator
from tqdm import tqdm
import argparse
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats


class SeedGenerator:
    """
    Auxiliary class for generating a series of random seed
    starting from a single seed. It requires the number of
    experiments k, and extract a seed from a uniform distribution.
    """
    def __init__(
            self,
            seed: int, # the seed for the generator
            k: int):   # the range for the uniform distribution
        self.k = k
        self.generator = np.random.default_rng(seed=seed)

    def __call__(self) -> int:
        return self.generator.integers(self.k, self.k**3)


def problem1(args, seed_generator: SeedGenerator) -> None:
    print('Problem 1')
    simulator = ConflictSimulator(
        m=None, # it is not needed to fix the population 
                # size in the first problem
        k=args.k1,
        distribution=args.distribution,
        seed=seed_generator(),
        verbose=True
    )
    avg_value = simulator.exec_sim_1()
    print(f'The average size of samples to have a collision is: {avg_value}')
    conf_int = scipy.stats.norm.interval(
        args.confidence,
        avg_value,
        np.sqrt(np.var(simulator.sim_1_samples)/args.k1)
    )
    print(f'The {args.confidence} confidence interval is: [{conf_int[0]}, {conf_int[1]}]')
    values, counts = np.unique(simulator.sim_1_samples, return_counts=True)
    counts = counts/args.k1
    max_value = values[np.argmax(counts)]
    _, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.bar(values, counts)
    ax.axvline(x=max_value, color='red')
    ax.set_title('Estimated PDF of the number of people needed to have a birthday collision')
    ax.set_xlabel('Population size [m]')
    ax.set_ylabel('P(X=m)')
    _, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.plot(np.cumsum(counts))
    ax.set_title('Estimated CDF of the number of people needed to have a birthday collision')
    ax.set_xlabel('Population size [m]')
    ax.set_ylabel('P(X<m)')


def problem2(args, seed_generator: SeedGenerator) -> None:
    print('Problem 2')
    m_values = np.arange(
        start=args.start,
        stop=args.stop+args.step,
        step=args.step,
        dtype=int
    )
    estimated_prob = np.empty_like(m_values, dtype=np.float64)
    lower_conf_int = np.empty_like(m_values, dtype=np.float64)
    upper_conf_int = np.empty_like(m_values, dtype=np.float64)
    limit = np.empty_like(m_values, dtype=np.float64)
    for i in tqdm(range(len(m_values))):
        m = m_values[i]
        simulator = ConflictSimulator(
            m=m,
            k=args.k2,
            seed=seed_generator(),
            distribution=args.distribution,
            verbose=False
        )
        p = simulator.exec_sim_2()
        estimated_prob[i] = p
        z = scipy.stats.norm.ppf(q=(1-args.confidence)/2)
        s_hat = np.sqrt(p*(1-p)/args.k2)
        lower_conf_int[i] = p - z*s_hat
        upper_conf_int[i] = p + z*s_hat
        limit[i] = 1 - np.exp(-m**2/730)
    _, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.plot(m_values, estimated_prob, color='red', marker='.')
    ax.plot(m_values, limit, color='black', linestyle='dashed')
    ax.fill_between(
        x=m_values,
        y1=lower_conf_int,
        y2=upper_conf_int,
        color='lightblue'
        )
    ax.legend(
        ['estimated probability', 'theoretical limit', f'{args.confidence} confidence interval'],
        loc='lower right'
        )
    ax.set_title('Probability of a collision w.r.t. size of the population')
    ax.set_xlabel('Size of the population [m]')
    ax.set_ylabel('Probability of birthday collision')


def main(args):
    seed_generator = SeedGenerator(
        seed=args.seed,
        k = max(args.k1, args.k2)
    )
    problem1(args, seed_generator)
    problem2(args, seed_generator)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k1',
        type=int,
        help='The number of experiments for the first problem'
    )
    parser.add_argument(
        '--k2',
        type=int,
        help='The number of experiments for the second problem'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='The number of people in each experiment in the second problem'
    )
    parser.add_argument(
        '--stop',
        type=int,
        help='The maximum number of people in each experiment in the second problem'
    )
    parser.add_argument(
        '--step',
        type=int,
        help='The step size for increasing the number of people in the second problem'
    )
    parser.add_argument(
        '--distribution',
        type=str,
        default='uniform',
        help='The distribution for generating birthdays,\
            a string in ["uniform", "realistic"]'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        help='Percentage of confidence interval (e.g., 0.95, 0.99)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for the simulation'
    )
    main(parser.parse_args())
