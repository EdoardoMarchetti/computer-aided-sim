from simulator import MultiServerSimulator
import numpy as np
import argparse
import atexit
import matplotlib.pylab as plt
from tqdm import tqdm


class SeedGenerator:
    def __init__(self, seed: int):
        self.generator = np.random.default_rng(seed=seed)

    def __call__(self) -> int:
        return self.generator.integers(low=100, high=10000)


def main(args):
    MARKERS = {
        'exp': 'x',
        'det': 'o',
        'hyp': '+'
    }
    atexit.register(lambda: plt.close('all'))
    get_seed = SeedGenerator(seed=args.seed)
    #inter_arrival_lambdas = (0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 2.8,)
    inter_arrival_lambdas = (0.2, 0.8, 1.4, 2.8)
    service_time_cases = ('a',)
    service_time_distributions = ('exp', 'det', 'hyp',)
    params_combinations = list()
    for i in range(len(inter_arrival_lambdas)):
        inter_arrival_lambda = inter_arrival_lambdas[i]
        for j in range(len(service_time_cases)):
            service_time_case = service_time_cases[j]
            for k in range(len(service_time_distributions)):
                service_time_distribution = service_time_distributions[k]
                params_combinations.append((
                    service_time_case,
                    service_time_distribution,
                    inter_arrival_lambda,
                ))
    for i in tqdm(range(
        len(service_time_cases)\
            *len(service_time_distributions)\
                *len(inter_arrival_lambdas))):
        service_time_case, \
            service_time_distribution, \
                inter_arrival_lambda = params_combinations[i]
        simulator = MultiServerSimulator(
            n_servers = args.n_servers,
            queue_size=args.queue_size,
            service_time_distribution=service_time_distribution,
            inter_arrival_lp_lambda=inter_arrival_lambda,
            inter_arrival_hp_lambda=inter_arrival_lambda,
            service_time_case=service_time_case,
            steady_batch_size=args.steady_batch_size,
            transient_batch_size=args.transient_batch_size,
            transient_tolerance=args.transient_tolerance,
            confidence=args.confidence,
            max_served_clients=args.max_served_clients,
            seed=get_seed()
            )
        values, means, n_batches = simulator.execute()
        _, ax = plt.subplots(1,1, figsize=(7,7))
        ax.plot(values['delay'])
        ax.plot(means['delay'])
    plt.show(block=False)
    input('Press enter to close all the figures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_servers',
        type=int,
        default=2,
        help='The number of servers.'
    )
    parser.add_argument(
        '--queue_size',
        type=int,
        default=1000,
        help='Maximum number of clients in the queue.'
    )
    parser.add_argument(
        '--transient_batch_size',
        type=int,
        default=1000,
        help='Batch size for identifying the transient state.'
    )
    parser.add_argument(
        '--steady_batch_size',
        type=int,
        default=1000,
        help='Batch size for batch means algorithm.'
    )
    parser.add_argument(
        '--transient_tolerance',
        type=float,
        default=1e-4,
        help='Tolerance for ending the batch means algorithm.'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=.95,
        help='Confidence interval (default .95).'
    )
    parser.add_argument(
        '--max_served_clients',
        type=int,
        default=10000,
        help='Accuracy level to be reached.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The seed for the random generator.'
    )
    main(parser.parse_args())
