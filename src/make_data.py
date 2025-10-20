import random


def generate_shuff_dyck(k, max_length=2048, p_open=0.5, max_depth=16):
    """
    Generate a k-shuffle Dyck sequence, truncated at max_length.
    When max depth is reached, close one bracket and continue.

    Args:
        k (int): Number of different types of brackets
        max_length (int): Target maximum length of the sequence
        p_open (float): Probability of opening a new bracket
        max_depth (int): Maximum nesting depth allowed

    Returns:
        list: Generated sequence where i represents opening bracket i
        and i+k represents closing bracket i

    Note: the final Dyck word may be invalid due to truncation, but
    we didn’t find this to be an issue in practice.
    """
    sequence = []
    counts = [0] * k  # Track open brackets of each type

    while len(sequence) < max_length:
        depth = sum(counts)

        # Must open if all brackets are closed
        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        # If at max depth, force a close
        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        # Randomly choose to open or close
        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            # Close an existing bracket
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def generate_sequence_wrapper(args_tuple):
    """Wrapper function for multiprocessing that unpacks arguments."""
    k, max_length, p_open, max_depth = args_tuple
    return generate_shuff_dyck(k, max_length, p_open, max_depth)


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool, cpu_count

    import numpy as np
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Generate k-shuffle Dyck sequences and save to .npy file."
    )
    parser.add_argument("n", type=int, help="Number of sequences to generate")
    parser.add_argument("output", type=str, help="Output .npy filename")
    parser.add_argument("--k", type=int, default=2, help="Number of bracket types")
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--p_open", type=float, default=0.5, help="Probability of opening a bracket"
    )
    parser.add_argument(
        "--max_depth", type=int, default=16, help="Maximum nesting depth"
    )
    parser.add_argument(
        "--workers", type=int, default=cpu_count(), help="Number of worker processes"
    )
    args = parser.parse_args()

    # Create argument tuples for each worker
    work_args = [(args.k, args.max_length, args.p_open, args.max_depth)] * args.n

    # Generate sequences in parallel
    with Pool(args.workers) as pool:
        sequences = list(
            tqdm(
                pool.imap(generate_sequence_wrapper, work_args),
                total=args.n,
                desc="Generating sequences",
            )
        )

    # Concatenate all sequences into one long sequence
    full_sequence = []
    for seq in sequences:
        full_sequence.extend(seq)

    # Save as a single long sequence
    np.save(args.output, np.array(full_sequence))
