import heapq

import numpy as np


def build_min_heap(freqs):
    '''Returns a min-heap of (frequency, token_index).'''
    freq_index = [(freq, i) for i, freq in enumerate(freqs)]
    # O(n log n) where n = len(freqs)
    heapq.heapify(freq_index)
    return freq_index


def huffman_tree(heap):
    '''Returns the Huffman tree given a min-heap of indices and frequencies.'''
    # Runs for n iterations where n = len(heap)
    while len(heap) > 1:
        # Remove the smallest two nodes. O(log n)
        freq1, ind1 = heapq.heappop(heap)
        freq2, ind2 = heapq.heappop(heap)
        # Create a parent node for these two nodes
        parent_freq = freq1 + freq2
        # The left child is the one with the lowest frequency
        parent_ind = (ind1, ind2)
        # Insert this parent node. O(log n)
        # FIXME in the rare case of equal frequencies, tuple comparison will fail
        # TODO add an second element in tuple for tiebreaking
        heapq.heappush(heap, (parent_freq, parent_ind))
    code_tree = heap[0][1]
    # Total runtime O(n log n).
    return code_tree


def tv_huffman(code_tree, p):
    '''Returns the total variation between a distribution over tokens
    and the distribution induced by a Huffman coding of the tokens.
    Args:
        code_tree : tuple.
            Huffman codes as represented by a binary tree.
        p : array of size of the vocabulary.
            The distribution over tokens.
    '''
    tot_l1 = 0
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
            tot_l1 += abs(p[node] - 2 ** (-depth))
            # tot_ce += p[node] * depth
    # Returns total variation
    return 0.5 * tot_l1


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
            code = codes.pop(0)
            if code == 'l':
                # Go left
                state = state[0]
            else:
                # Go right
                state = state[1]
        else:
            # A leaf node, decode a letter
            decoded.append(state)
            # Reset state
            state = code_tree
    return decoded


if __name__ == '__main__':
    v = 100
    p = np.random.dirichlet([1] * v)
    print(sum(p))
    # p = [0.8, 0.05, 0.11, 0.049]
    heap = build_min_heap(p)
    print(heap)

    tree = huffman_tree(heap)
    print(tree)
    # print(huffman_tree(build_min_heap([1, 1, 1, 1, 1, 1])))
    print(tv_huffman(tree, p))
    print(invert_code_tree(tree))

    string = np.random.choice(v, 10, p=p)
    print(list(string))
    codes = encode(tree, string)
    print(codes)
    print(decode(tree, codes))
