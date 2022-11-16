from simulator import QueueSimulator
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


class SeedGenerator:
    def __init__(self, seed: int):
        self.generator = np.random.default_rng(seed=seed)

    def __call__(self) -> int:
        return self.generator.integers(low=100, high=10000)


def main(args):
    if args.transient_batch_size is None:
        args.transient_batch_size = args.steady_batch_size
    utilisations = (args.utilisation,) \
        if args.utilisation is not None else \
        (0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99, )
    service_distributions = (args.service_distribution, ) \
        if args.service_distribution is not None else \
            ('exp', 'det', 'hyp', )
    seed_generator = SeedGenerator(seed=args.seed)
    for service_distribution in service_distributions:
        for utilisation in utilisations:
            sim = QueueSimulator(
                utilisation=utilisation,
                service_distribution=service_distribution,
                steady_batch_size=args.steady_batch_size,
                transient_batch_size=args.transient_batch_size,
                transient_tolerance=args.transient_tolerance,
                confidence=args.confidence,
                seed=seed_generator(),
                verbose=args.verbose
            )
            print(sim)
            mean, conf_int = sim.exec(collect='departure')
            _, ax = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(15, 10)
                )
            if args.verbose:
                ax[0].plot(sim.delays)
                ax[0].plot(sim.cumulative_means)
                ax[0].axvline(x=sim.transient_end, color='red')
                ax[0].set_title('Delay')
                ax[0].set_xlabel('Time')
                ax[0].set_ylabel('Delay of a client at departure time')
                
                ax[1].plot(sim.queue_sizes)
                ax[1].plot(sim.cumulative_means)
                ax[1].axvline(x=sim.transient_end, color='red')
                ax[1].set_title('Queue size')
                ax[1].set_xlabel('Time')
                ax[1].set_ylabel('Number of clients in the queue')
            
            #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--utilisation',
        type=float,
        help='Utilisation factor of the server.'
    )
    parser.add_argument(
        '--service_distribution',
        type=str,
        help='A string in ["exp", "det", "hyp"],\
            the distribution of the service time.'
    )
    parser.add_argument(
        '--transient_batch_size',
        type=int,
        default=100,
        help='Batch size for identifying the transient state.'
    )
    parser.add_argument(
        '--steady_batch_size',
        type=int,
        default=100,
        help='Batch size for batch means algorithm.'
    )
    parser.add_argument(
        '--transient_tolerance',
        type=float,
        default=1e-6,
        help='Tolerance for ending the batch means algorithm.'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=.95,
        help='Confidence interval (default .95).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The seed for the random generator.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Use this flag to print informations\
            during the simulation.'
    )
    main(parser.parse_args())
