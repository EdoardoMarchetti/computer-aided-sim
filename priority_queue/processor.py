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


def plot_graph(
        service_time_case: str,
        inter_arrival_lambda: float,
        service_time_distribution: str,
        delay, left_conf_int, right_conf_int,
        hp_delay, hp_left_conf_int, hp_right_conf_int,
        lp_delay, lp_left_conf_int, lp_right_conf_int) -> None:
    TO_PLOT = set()
    TO_PLOT.add(('b', 0.2, 'exp'))
    TO_PLOT.add(('b', 0.2, 'hyp'))
    TO_PLOT.add(('a', 0.4, 'hyp'))
    TO_PLOT.add(('a', 0.8, 'det'))
    TO_PLOT.add(('a', 1.4, 'hyp'))
    TO_PLOT.add(('b', 1.4, 'hyp'))
    params = (service_time_case, inter_arrival_lambda, service_time_distribution,)
    if params in TO_PLOT:
        _, ax = plt.subplots(1, 1, figsize=(7,7))
        ax.fill_between(
            x=np.arange(len(delay)),
            y1=left_conf_int,
            y2=right_conf_int,
            )
        ax.fill_between(
            x=np.arange(len(hp_delay)),
            y1=hp_left_conf_int,
            y2=hp_right_conf_int,
        )
        ax.fill_between(
            x=np.arange(len(lp_delay)),
            y1=lp_left_conf_int,
            y2=lp_right_conf_int,
        )
        ax.scatter(np.arange(len(delay)), delay)
        ax.scatter(np.arange(len(hp_delay)), hp_delay)
        ax.scatter(np.arange(len(lp_delay)), lp_delay)
        title = 'Average delay\n' + \
            f'Service time case: {service_time_case}\n' + \
            f'Service time distribution: {service_time_distribution}\n' + \
            f'Inter arrival lambda: {inter_arrival_lambda}'
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average delay')
        ax.legend((
            'Aggregate', 'Aggregate .95 conf int',
            'High priority', 'High priority .95 conf int',
            'Low priority', 'Low priority .95 conf int'))


def main(args):
    atexit.register(lambda: plt.close('all'))
    get_seed = SeedGenerator(seed=args.seed)
    inter_arrival_lambdas = (0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 2.8,)
    service_time_cases = ('a', 'b')
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
    mean_accuracy = 0
    for i in tqdm(range(len(params_combinations))):
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
        results = simulator.execute()
        means = results['means']
        delay, left_conf_int, right_conf_int = \
            means['mean_delay'][:,0], means['mean_delay'][:,1], means['mean_delay'][:,2]
        hp_delay, hp_left_conf_int, hp_right_conf_int = \
            means['mean_delay_hp'][:,0], means['mean_delay_hp'][:,1], means['mean_delay_hp'][:,2]
        lp_delay, lp_left_conf_int, lp_right_conf_int = \
            means['mean_delay_lp'][:,0], means['mean_delay_lp'][:,1], means['mean_delay_lp'][:,2]
        plot_graph(
            service_time_case,
            inter_arrival_lambda,
            service_time_distribution,
            delay, left_conf_int, right_conf_int,
            hp_delay, hp_left_conf_int, hp_right_conf_int,
            lp_delay, lp_left_conf_int, lp_right_conf_int
            )
        accuracy = results['accuracy']
        mean_accuracy += accuracy['mean_delay']
    mean_accuracy = mean_accuracy / len(params_combinations)
    print(f'Mean accuracy: {mean_accuracy}')
    plt.show(block=False)
    input('Press enter to close all the figures')


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
