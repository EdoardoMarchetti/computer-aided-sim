from simulator import AntiPlagiarismSimulator
import argparse
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def main(args):
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    legend = []
    window_size = (4, 8,)
    for S in window_size:
        print(f'Window size S: {S}')
        legend.append(f'Sentences set size, S: {S}')
        legend.append(f'Theoretical sentences set size, S: {S}')
        legend.append(f'Hashes set size, S: {S}')
        legend.append(f'Theoretical hashes set size, S: {S}')
        tolerances = []
        sentence_set_sizes = []
        hash_set_sizes = []
        theoretical_sentence_set_size = []
        theoretical_hash_set_size = []
        fp_tolerance = 0.1
        for _ in tqdm(range(6)):
            tolerances.append(fp_tolerance)
            simulator = AntiPlagiarismSimulator(
                filepath=args.filepath,
                window_size=S,
                fp_tolerance=fp_tolerance
                )
            simulator.process()
            theoretical_size = lambda m: np.ceil(m*np.log2(m/fp_tolerance))
            theoretical_sentence_set_size.append(
                theoretical_size(len(simulator.distinct_sentences))
                )
            theoretical_hash_set_size.append(
                theoretical_size(len(simulator.distinct_hash_sentences))
                    )
            sentence_set_sizes.append(sys.getsizeof(
                simulator.distinct_sentences
                ))
            hash_set_sizes.append(sys.getsizeof(
                simulator.distinct_hash_sentences
                ))
            fp_tolerance /= 10.0
        n_sentences = len(simulator.distinct_sentences)
        print(f'For S={S} there are {n_sentences} stored.')
        n_hashes = len(simulator.distinct_hash_sentences)
        print(f'')
        ax.set_xscale('log')
        ax.plot(tolerances, sentence_set_sizes)
        ax.scatter(tolerances, theoretical_sentence_set_size)
        ax.plot(tolerances, hash_set_sizes)
        ax.scatter(tolerances, theoretical_hash_set_size)
    ax.legend(legend)
    ax.set_ylabel('Bytes')
    ax.set_xlabel('PR(False Positive)')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filepath',
        type=str,
        required=True,
        help='Path to the file containing text for anti-plagiarism.'
    )
    main(parser.parse_args())
