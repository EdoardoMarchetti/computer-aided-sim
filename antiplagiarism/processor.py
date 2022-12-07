from simulator import AntiPlagiarismSimulator
import argparse
from pympler import asizeof
from matplotlib import pyplot as plt
from tqdm import tqdm


def main(args):
    fp_tolerance = 0.1
    tolerances = []
    sentence_set_sizes = []
    hash_set_sizes = []
    for _ in tqdm(range(6)):
        tolerances.append(fp_tolerance)
        simulator = AntiPlagiarismSimulator(
            filepath=args.filepath,
            window_size=args.window_size,
            fp_tolerance=fp_tolerance
            )
        simulator.process()
        sentence_set_sizes.append(asizeof.asizeof(
            simulator.distinct_sentences
            ))
        hash_set_sizes.append(asizeof.asizeof(
            simulator.distinct_hash_sentences
            ))
        fp_tolerance /= 10.0
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.set_xscale('log')
    ax.plot(tolerances, sentence_set_sizes)
    ax.plot(tolerances, hash_set_sizes)
    ax.legend(('Sentences set size', 'Hashes set size',))
    ax.set_ylabel('Bytes')
    ax.set_xlabel('PR(False Positive)')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filepath',
        type=str,
        default='commedia.txt',
        help='Path to the file containing text for anti-plagiarism.'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=4,
        help='Size of the sentences.'
    )
    main(parser.parse_args())
