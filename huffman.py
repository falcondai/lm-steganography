import heapq

import numpy as np


def build_min_heap(freqs, inds=None):
    '''Returns a min-heap of (frequency, token_index).'''
    inds = inds or range(len(freqs))
    # Add a counter in tuples for tiebreaking
    freq_index = [(freqs[ind], i, ind) for i, ind in enumerate(inds)]
    # O(n log n) where n = len(freqs)
    heapq.heapify(freq_index)
    return freq_index


def huffman_tree(heap):
    '''Returns the Huffman tree given a min-heap of indices and frequencies.'''
    # Add a counter in tuples for tiebreaking
    t = len(heap)
    # Runs for n iterations where n = len(heap)
    while len(heap) > 1:
        # Remove the smallest two nodes. O(log n)
        freq1, i1, ind1 = heapq.heappop(heap)
        freq2, i2, ind2 = heapq.heappop(heap)
        # Create a parent node for these two nodes
        parent_freq = freq1 + freq2
        # The left child is the one with the lowest frequency
        parent_ind = (ind1, ind2)
        # Insert this parent node. O(log n)
        heapq.heappush(heap, (parent_freq, t, parent_ind))
        t += 1
    code_tree = heap[0][2]
    # Total runtime O(n log n).
    return code_tree


def tv_huffman(code_tree, p):
    '''Returns the total variation between a distribution over tokens and the
    distribution induced by a Huffman coding of (a subset of) the tokens.

    Args:
        code_tree : tuple.
            Huffman codes as represented by a binary tree. It might miss some
            tokens.
        p : array of size of the vocabulary.
            The distribution over all tokens.
    '''
    tot_l1 = 0
    # The tokens absent in the Huffman codes have probability 0
    absence = np.ones_like(p)
    # tot_ce = 0
    # Iterate leaves of the code tree. O(n)
    stack = []
    # Push the root and its depth onto the stack
    stack.append((code_tree, 0))
    while len(stack) > 0:
        node, depth = stack.pop()
        if type(node) is tuple:
            # Expand the children
            left_child, right_child = node
            # Push the children and their depths onto the stack
            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))
        else:
            # A leaf node
            ind = node
            tot_l1 += abs(p[ind] - 2 ** (-depth))
            absence[ind] = 0
            # tot_ce += p[ind] * depth
    # Returns total variation
    print('abs', absence, absence * p, tot_l1)
    return 0.5 * (tot_l1 + np.sum(absence * p))


def total_variation(p, q):
    '''Returns the total variation of two distributions over a finite set.'''
    # We use 1-norm to compute total variation.
    # d_TV(p, q) := sup_{A \in sigma} |p(A) - q(A)|
    # = 1/2 * sum_{x \in X} |p(x) - q(x)| = 1/2 * ||p - q||_1
    return 0.5 * np.sum(np.abs(p - q))


def invert_code_tree(code_tree):
    '''Build a map from letters to codes'''
    code = dict()
    stack = []
    stack.append((code_tree, ''))
    while len(stack) > 0:
        node, code_prefix = stack.pop()
        if type(node) is tuple:
            left, right = node
            stack.append((left, code_prefix + 'l'))
            stack.append((right, code_prefix + 'r'))
        else:
            code[node] = code_prefix
    return code


def encode(code_tree, string):
    '''Encode a string with a given Huffman coding.'''
    code = invert_code_tree(code_tree)
    encoded = ''
    for letter in string:
        encoded += code[letter]
    return encoded


def decode(code_tree, encoded):
    '''Decode an Huffman-encoded string.'''
    decoded = []
    state = code_tree
    codes = [code for code in encoded]
    # Terminate when there are no more codes and decoder state is resetted
    while not (len(codes) == 0 and type(state) is tuple):
        if type(state) is tuple:
            # An internal node
            left, right = state
            try:
                code = codes.pop(0)
            except IndexError:
                raise Exception('Decoder should stop at the end of the encoded string. The string may not be encoded by the specified Huffman coding.')
            if code == 'l':
                # Go left
                state = left
            else:
                # Go right
                state = right
        else:
            # A leaf node, decode a letter
            decoded.append(state)
            # Reset decoder state
            state = code_tree
    return decoded


if __name__ == '__main__':
    v = 5
    p = np.random.dirichlet([1] * v)
    print(sum(p))
    p = [0.7, 0.1, 0.05, 0.1, 0.05]
    heap = build_min_heap(p, [0, 1, 2, 4])
    print(heap)

    tree = huffman_tree(heap)
    print(tree)
    print(tv_huffman(tree, p))
    print(invert_code_tree(tree))

    # string = np.random.choice(v, 10, p=p)
    string = [0, 0, 2, 4, 1, 0, 2, 2]
    print(list(string))
    codes = encode(tree, string)
    print(codes)
    print(decode(tree, codes))
