from simulator import AntiPlagiarismSimulator
import argparse
from pympler.asizeof import asizeof
from matplotlib import pyplot as plt
import numpy as np
import atexit


def main(args):
    atexit.register(lambda: plt.close('all'))
    _, ax_sizes = plt.subplots(1, 1, figsize=(7,7))
    _, ax_theoretical = plt.subplots(1, 1, figsize=(7,7))
    legend_sizes = []
    legend_theoretical = []
    window_size = (4, 8,)
    for S in window_size:
        tolerances = []
        sentence_set_sizes = []
        hash_set_sizes = []
        theoretical_sentence_set_size = []
        theoretical_hash_set_size = []
        fp_tolerance = 0.1
        for _ in range(5):
            tolerances.append(fp_tolerance)
            simulator = AntiPlagiarismSimulator(
                filepath=args.filepath,
                window_size=S,
                fp_tolerance=fp_tolerance
                )
            simulator.process()
            theoretical_size = lambda m: 8*np.ceil(np.log2(m/fp_tolerance))
            n_sentences = len(simulator.distinct_sentences)
            theoretical_sentence_set_size.append(
                theoretical_size(n_sentences)
                )
            n_hashes = len(simulator.distinct_hash_sentences)
            theoretical_hash_set_size.append(
                theoretical_size(n_hashes)
                    )
            sentence_set_sizes.append(asizeof(
                simulator.distinct_sentences
                ))
            hash_set_sizes.append(asizeof(
                simulator.distinct_hash_sentences
                ))
            print(f'For S={S}, fp_tol={fp_tolerance}, the set of sentences theoretical size is {theoretical_sentence_set_size[-1]} B')
            print(f'For S={S}, fp_tol={fp_tolerance}, the set of hashes theoretical size is {theoretical_hash_set_size[-1]} B')
            print(f'For S={S}, fp_tol={fp_tolerance}, there are {n_sentences} distinct sentences stored.')
            print(f'For S={S}, fp_tol={fp_tolerance}, there are {n_hashes} distinct hashes stored.')
            print()
            fp_tolerance = fp_tolerance / 10
        ax_sizes.plot(tolerances, sentence_set_sizes)
        ax_theoretical.scatter(tolerances, theoretical_sentence_set_size)
        ax_sizes.plot(tolerances, hash_set_sizes)
        ax_theoretical.scatter(tolerances, theoretical_hash_set_size)
        legend_sizes.append(f'Sentences set size, S: {S}')
        legend_theoretical.append(f'Theoretical sentences set size, S: {S}')
        legend_sizes.append(f'Hashes set size, S: {S}')
        legend_theoretical.append(f'Theoretical hashes set size, S: {S}')
    ax_sizes.set_xscale('log')
    ax_theoretical.set_xscale('log')
    ax_sizes.legend(legend_sizes)
    ax_theoretical.legend(legend_theoretical)
    ax_theoretical.set_ylabel('Bytes')
    ax_sizes.set_ylabel('Bytes')
    ax_theoretical.set_xlabel('PR(False Positive)')
    ax_sizes.set_xlabel('PR(False Positive)')
    ax_theoretical.set_title('Theoretical set size')
    ax_sizes.set_title('Effective set size')
    plt.show(block=False)
    input('Press enter to close all the figures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filepath',
        type=str,
        required=True,
        help='Path to the file containing text for anti-plagiarism.'
    )
    main(parser.parse_args())
