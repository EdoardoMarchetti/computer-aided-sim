from simulator import GWSimulator
from matplotlib import pyplot as plt
import argparse
import numpy as np
from scipy import stats
import atexit


class SeedGenerator:
    def __init__(
            self,
            k: int,
            seed: int):
        self.generator = np.random.default_rng(seed=seed)
        self.low = k
        self.high = k**3
    
    def __call__(self) -> int:
        return self.generator.integers(
            low=self.low,
            high=self.high
            )


def prob_confidence_interval(
        p: float,
        confidence: float,
        k: int
        ) -> tuple[float, float]:
    z = stats.norm.ppf(q=(1-confidence)/2)
    s_hat = np.sqrt(p*(1-p)/k)
    return (p - z*s_hat, p + z*s_hat)


def MGF(n: int, lam: float) -> np.ndarray:
    mgf = np.empty(shape=(n,), dtype=float)
    mgf[0] = np.exp(-lam)
    for i in range(1, n):
        mgf[i] = np.exp(lam*(mgf[i-1]-1))
    return mgf


def main(args):
    atexit.register(lambda: plt.close('all'))
    lams = (0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.3,)
    get_seed = SeedGenerator(
        seed=args.seed,
        k = args.k
        )
    for lam in lams:
        print(f'Lambda: {lam}')
        death_gens = np.empty(shape=(args.k,), dtype=int)
        infinite_trees = 0
        n_death = 0
        n_nodes = np.empty(shape=(args.k,), dtype=int)
        for i in range(args.k):
            simulator = GWSimulator(
                lam=lam,
                max_gen_nodes=args.max_gen_nodes,
                seed = get_seed()
                )
            death_gen = simulator.execute()
            n_nodes[i] = simulator.total_nodes
            if death_gen >= 0:
                death_gens[n_death] = death_gen
                n_death += 1
            else:
                infinite_trees += 1
        death_gens = death_gens[:n_death]
        generations, counts = np.unique(death_gens, return_counts=True)
        probs = counts / args.k
        prob_infinite = infinite_trees / args.k
        print(f'Prob. infinite tree: {prob_infinite}')
        conf_int = prob_confidence_interval(
            p=prob_infinite,
            confidence=args.confidence,
            k=args.k
            )
        print(f'{args.confidence} interval: [{conf_int[1]}, {conf_int[0]}]')
        probs = np.cumsum(probs)
        conf_int_left = np.empty(shape=(len(probs),), dtype=float)
        conf_int_right = np.empty(shape=(len(probs),), dtype=float)
        for i in range(len(probs)):
            conf_int_right[i], conf_int_left[i] = \
                prob_confidence_interval(
                    p=probs[i],
                    confidence=args.confidence,
                    k=args.k
                    )
        max_gen = np.max(generations)
        mgf = MGF(n=max_gen, lam=lam)
        _, ax = plt.subplots(1, 1, figsize=(7,7))
        ax.fill_between(
            x=generations,
            y1=conf_int_left,
            y2=conf_int_right,
            color='lightblue'
            )
        ax.plot(mgf, color='red', linestyle='dashed')
        ax.scatter(generations, probs, marker='x', color='black')
        ax.set_xlabel('Generation i')
        ax.set_ylabel('Prob. of extinction before generation i')
        ax.set_title(f'Lambda: {lam}')
        ax.legend(
            (
                f'{args.confidence} confidence interval',
                'Theoretical prob. of extinction',
                'Estimated prob. of extinction',
            )
        )
        if lam == 0.8:
            _, ax = plt.subplots(1, 1, figsize=(7,7))
            ax.hist(n_nodes)
            ax.set_xlabel('Number of nodes')
            ax.set_ylabel('Number of trees')
            ax.set_title('Histogram of the number of nodes in the tree.')
    plt.show(block=False)
    input('Press enter to close figures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k',
        type=int,
        default=10000,
        help='Number of experiments.'
    )
    parser.add_argument(
        '--max_gen_nodes',
        type=int,
        default=15,
        help='Maximum number of nodes in the same generation for killing potentially infinite GW trees.'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Confidence interval.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the generator.'
    )
    main(parser.parse_args())
