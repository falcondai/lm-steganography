
import numpy as np

twitter_most_frequent_words = ['the','i', 'to', 'a', 'and', 'is', 'in', 'it', 'you', 'of']

def softmax(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def total_variation(p, q):
    '''Returns the total variation of two distributions over a finite set.'''
    # We use 1-norm to compute total variation.
    # d_TV(p, q) := sup_{A \in sigma} |p(A) - q(A)|
    # = 1/2 * sum_{x \in X} |p(x) - q(x)| = 1/2 * ||p - q||_1
    return 0.5 * np.sum(np.abs(p - q))


def bitblock_to_tokens(vocab, N_blocks_length):
    '''
    Args:
        vocab: vocabulary in a list [token1, token2, ..., tokenx]
        N_blocks: number of bit blocks
    Return:
        bits2tokens: a dictionary {bit_block: token_list}
        token2bits: a dictionary {token: it's bit_block}
    '''
    assert N_blocks_length < 8 # 2^10 is too large, each block contains too few tokens

    N_blocks = 2 ** N_blocks_length
    batch_size = len(vocab) // N_blocks

    bits2tokens = {}
    token2bits = {}

    for i in range(N_blocks-1):
        format_str = '0'+str(N_blocks_length)+'b'
        bits = format(i, format_str)
        bits2tokens[bits] = vocab[i*batch_size: (i+1)*batch_size]
    last_format_str = '0'+str(N_blocks_length)+'b'
    last_bits = format(N_blocks - 1, last_format_str)
    bits2tokens[last_bits] = vocab[(N_blocks-1)*batch_size:]
    for key, tokens in bits2tokens.items():
        for token in tokens:
            token2bits[token] = key

    return bits2tokens, token2bits

def encode(bits2tokens, N_blocks_length, bitstring, p):
    '''
    Args:
        bits2tokens_common: a dictionary {bits: token_ls}
        bitstring: the bit string
        p: probability distribution over vocab
    Return: 
        stega_text
    '''

    blocks = 2**N_blocks_length
    blocks_num = len(bitstring) // N_blocks_length

    # get bits block sequence
    bits_blocks = [bitstring[i*N_blocks_length: (i+1)*N_blocks_length] for i in range(blocks_num)]

    # get the decimal representation of those bits
    decimal_rep = [int(bit_block, 2) for bit_block in bits_blocks]

    # get block size on p distribution
    p_blocks_size = p.shape[0] // blocks
    
    # divide p into blocks
    block_p = []

    for i in range(blocks-1):
        block_p.append(p[i*p_blocks_size: (i+1)*p_blocks_size])
    block_p.append(p[(blocks-1)*p_blocks_size: ])

    # get the p blocks corresponding to the bits blocks
    correspond_p_blocks = [block_p[i] for i in decimal_rep]

    # re-normalize the p block
    normed_block_p = [softmax(p) for p in correspond_p_blocks]

    tokens = []
    for i in range(blocks_num):
        block_tokens = bits2tokens[bits_blocks[i]]
        
        token = np.random.choice(block_tokens, 1, replace=False, p=normed_block_p[i])
        tokens.append(token[0])

        while token in twitter_most_frequent_words:
            token = np.random.choice(block_tokens, 1, replace=False, p=normed_block_p[i])
            tokens.append(token[0])
    
    stega_text = " ".join(tokens)
    return stega_text


def decode(token2bits, N_blocks_length, string):
    '''
    Args:
        token2bits: a dictionary {token: bit_block}
        string: a sentence(assume can be tokenized by space)
    Return: 
        bitstring
    '''
    tokens = string.split(" ")
    encoded_ls = []
    for i in range(len(tokens)):
        if tokens[i] in twitter_most_frequent_words:
            continue
        else:
            bit_block = token2bits[tokens[i].lower()]
            encoded_ls.append(bit_block)

    bitstring = "".join(encoded_ls)
    return bitstring

def common_token_adder(bits2tokens, common_tokens):
    """
    Args:
        bits2tokens: dictionary {bits_block: [tokens]}
        common_tokens: a list of the most frequently used tokens
    Return:
        bits2tokens_common: a dictionary with each block been added the common_tokens
    """
    bits2tokens_common = {}
    for bits, tokens in bits2tokens.items():
        bits2tokens_common[bits] = tokens+common_tokens
    return bits2tokens_common

if __name__ == '__main__':
    vocabulary = ['chocolate', 'love', 'apple', 'blueberry', 'muffin', 'yogurt', 'banana', 'cheesecake', 'ice', 'cream']

    p = np.array([1.0/len(vocabulary) for _ in range(len(vocabulary))])

    test_bits = "000010111"
    N_bits = 3
    bits2tokens, token2bits = bitblock_to_tokens(vocabulary, N_bits)

    # print("-"*30+"dictionaries: "+"-"*30)
    # print(bits2tokens)
    # print(token2bits)
    # bits2tokens_common = common_token_adder(bits2tokens, twitter_most_frequent_words)

    stega_text = encode(bits2tokens, N_bits, test_bits, p)
    print("-"*30+"stega_text"+"-"*30)
    print(stega_text)
    print("-"*64)
    origin_bits = decode(token2bits, N_bits, stega_text)
    print("-"*30+"origin bits"+"-"*30)
    print(origin_bits)
    print("-"*71)

