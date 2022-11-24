from simulator import QueueSimulator
import argparse
from matplotlib import pyplot as plt
import numpy as np
import atexit
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
    if args.transient_batch_size is None:
        args.transient_batch_size = args.steady_batch_size
    utilisations = np.array((0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99, ))
    utilisations_plot = 100*utilisations
    generate_seed = SeedGenerator(seed=args.seed)
    transient_batches = np.empty_like(utilisations)
    steady_batches = np.empty_like(utilisations)
    _, mean_plot = plt.subplots(1, 1, figsize=(8,8))
    _, batch_plot = plt.subplots(1, 1, figsize=(8,8))
    for service_distribution in ('exp', 'det', 'hyp', ):
        print(service_distribution)
        mean = np.empty(shape=(len(utilisations)), dtype=np.float64)
        left_conf_int = np.empty_like(mean)
        right_conf_int = np.empty_like(mean)
        for i in tqdm(range(len(utilisations))):
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
            mean[i], \
            (left_conf_int[i], right_conf_int[i]), \
            transient_batches[i],\
            steady_batches[i] = \
                sim.exec(collect='departure')
            if utilisation == 0.8:
                _, delay_plot = plt.subplots(1, 1, figsize=(8, 8))
                delay_plot.plot(sim.delays, color='lightblue')
                delay_plot.plot(sim.cumulative_means, linestyle='dotted', linewidth=4)
                delay_plot.axvline(x=sim.transient_end, color='red', linestyle='dashed', linewidth=2)
                delay_plot.legend(('Delay', 'Cumulative mean delay', 'Transient end'))
                delay_plot.set_title(f'Delay\nutilisation: {utilisation}\nservice_distribution: {service_distribution}')
                delay_plot.set_xlabel('Time')
                delay_plot.set_ylabel('Delay')
        # end for
        batch_plot.plot(utilisations, steady_batches, marker='.')
        mean_plot.scatter(
            x=utilisations_plot,
            y=mean,
            marker=MARKERS[service_distribution],
            color='black',
            zorder=2
            )
        mean_plot.fill_between(
                x=utilisations_plot,
                y1=left_conf_int,
                y2=right_conf_int,
                zorder=1
                )
    # end for
    mean_plot.legend((
        'Mean delay exponential', f'{args.confidence} confidence interval exponential',
        'Mean delay deterministic', f'{args.confidence} confidence interval deterministic',
        'Mean delay hyperexponential2', f'{args.confidence} confidence interval hyperexponential2',
        ))
    mean_plot.set_xticks(utilisations_plot)
    mean_plot.set_title('Mean delay in function of the server utilisation')
    mean_plot.set_xlabel('Server utilisation level (%)')
    mean_plot.set_ylabel('Mean delay')
    batch_plot.legend((
        'Steady batches exp distribution',
        'Steady batches det distribution',
        'Steady batches hyp distribution',
    ))
    batch_plot.set_title('Number of batches needed for the simulation')
    batch_plot.set_xlabel('Server utilisation level')
    batch_plot.set_ylabel('Number of batches')
    plt.show(block=False)
    input('\nPress enter to close all the figures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
