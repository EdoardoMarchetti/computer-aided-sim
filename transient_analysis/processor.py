from simulator import QueueSimulator
import argparse
from matplotlib import pyplot as plt
import numpy as np
import atexit


class SeedGenerator:
    def __init__(self, seed: int):
        self.generator = np.random.default_rng(seed=seed)

    def __call__(self) -> int:
        return self.generator.integers(low=100, high=10000)


def main(args):
    atexit.register(lambda: plt.close('all'))
    if args.transient_batch_size is None:
        args.transient_batch_size = args.steady_batch_size
    utilisations = (args.utilisation,) \
        if args.utilisation is not None else \
        (0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99, )
    utilisations_plot = 100*np.array(utilisations)
    service_distributions = (args.service_distribution, ) \
        if args.service_distribution is not None else \
            ('exp', 'det', 'hyp', )
    generate_seed = SeedGenerator(seed=args.seed)
    for service_distribution in service_distributions:
        mean = np.empty(shape=(len(utilisations)), dtype=np.float64)
        left_conf_int = np.empty_like(mean)
        right_conf_int = np.empty_like(mean)
        for i in range(len(utilisations)):
            utilisation = utilisations[i]
            sim = QueueSimulator(
                utilisation=utilisation,
                service_distribution=service_distribution,
                steady_batch_size=args.steady_batch_size,
                transient_batch_size=args.transient_batch_size,
                transient_tolerance=args.transient_tolerance,
                confidence=args.confidence,
                accuracy=args.accuracy,
                seed=generate_seed()
            )
            print(sim)
            mean[i], (left_conf_int[i], right_conf_int[i]) = \
                 sim.exec(collect='departure')
            _, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.plot(sim.delays, color='lightblue')
            ax.plot(sim.cumulative_means, linestyle='dotted', linewidth=4)
            ax.axvline(x=sim.transient_end, color='red', linestyle='dashed', linewidth=2)
            ax.legend(('Delay', 'Mean delay', 'Transient end'))
            ax.set_title(f'Delay\nutilisation: {utilisation}\nservice_distribution: {service_distribution}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Delay')
        # end for
        _, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.plot(utilisations_plot, mean, color='black', marker='o')
        ax.fill_between(
                x=utilisations_plot,
                y1=left_conf_int,
                y2=right_conf_int,
                color='lightblue'
                )
        ax.legend(('Batch mean delay', f'{args.confidence} confidence interval'))
        ax.set_xticks(utilisations_plot)
        ax.set_title(f'Mean delay in function of the server utilisation\ndistribution: {service_distribution}')
        ax.set_xlabel('Server utilisation level (%)')
        ax.set_ylabel('Mean delay')
    # end for
    plt.show(block=False)
    input('\nPress enter to close all the figures')


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
        default=1000,
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
        '--accuracy',
        type=float,
        default=0.2,
        help='Accuracy level to be reached.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The seed for the random generator.'
    )
    main(parser.parse_args())
