import numpy as np

from huffman import build_min_heap, huffman_tree, tv_huffman, invert_code_tree


class LmAdversary:
    '''
    An adversary model that detects abnormal (cipher)texts based on an LM.
    '''

    def __init__(self, lm, detection_threshold):
        self.lm = lm
        self.detection_threshold = detection_threshold

    def is_abnormal(self, sentence):
        return self.detection_threshold < self.lm.perplexity(sentence)


class Sender:
    '''
    Sender model

    plain_text - `cipher` -> cipher_text - `hide` -> stego_text
    '''

    def __init__(self, lm, cipher_key, cipher, cipher_text_length, tv_threshold, max_sequence_length, seed=None):
        self.cipher_key = cipher_key
        self.cipher = cipher
        self.lm = lm
        self.tv_threshold = tv_threshold
        self.cipher_text_length = cipher_text_length
        self.random = np.random.RandomState(seed)
        self.acc_risk = 0
        self.max_sequence_length = max_sequence_length

    def encrypt(self, plain_text):
        cipher_text = self.cipher(self.cipher_key, plain_text)
        return cipher_text

    def hide(self, cipher_text):
        '''We use the cipher text to control the forward sampling procedure
        in sampling tokens from the prefix language model.'''
        assert len(cipher_text) == self.cipher_text_length, 'Ciphertext must be of length %s.' % self.cipher_text_length
        stego_text = []
        while len(cipher_text) > 0:
            inds = self.embed_bits(cipher_text)
            # Look up the tokens
            # XXX keep sequences separate
            stego_text.append([self.lm.vocabulary[ind] for ind in inds])
        return stego_text

    def embed_bits(self, coin_flips):
        '''We use a sequence of coin flips to control the generation of token
        indices from a language model. This returns _a sequence_ as defined by
        the language model, e.g. sentence, paragraph.'''
        ind = self.lm.SOS_ind
        prefix = [ind]
        p = self.lm.p_next_token(prefix)
        # Terminate the generation after we generate the EOS token
        while len(prefix) == 1 or (len(prefix) < self.max_sequence_length and ind != self.lm.EOS_ind):
            # There is still some cipher text to hide
            le = len(coin_flips)
            if le > 0:
                # Build Huffman codes for the conditional distribution
                heap = build_min_heap(p)
                hc = huffman_tree(heap)
                # Check if the total variation is low enough
                print(tv_huffman(hc, p))
                if tv_huffman(hc, p)[0] < self.tv_threshold:
                    # Huffman-decode the cipher text into a token
                    # Consume the cipher text until a token is generated
                    decoder_state = hc
                    while type(decoder_state) is tuple:
                        left, right = decoder_state
                        try:
                            bit = coin_flips.pop(0)
                        except IndexError:
                            # No more cipher text. Pad with random bits
                            bit = self.random.choice(2)
                        # 0 => left, 1 => right
                        decoder_state = left if bit == 0 else right
                    # Decoder settles in a leaf node
                    ind = decoder_state
                    prefix.append(ind)
                    p = self.lm.p_next_token(prefix)
                    continue
            # Forward sample according to LM normally
            ind = self.random.choice(self.lm.vocabulary_size, p=p)
            prefix.append(ind)
            p = self.lm.p_next_token(prefix)
        # Drop the EOS index
        return prefix[1:]


class Receiver:
    '''
    Receiver model

    stego_text - `seek` -> cipher_text - `decipher` -> plain_text
    '''

    def __init__(self, lm, decipher_key, decipher, cipher_text_length, tv_threshold):
        self.decipher_key = decipher_key
        self.decipher = decipher
        self.lm = lm
        self.cipher_text_length = cipher_text_length
        self.tv_threshold = tv_threshold

    def decrypt(self, cipher_text):
        plain_text = self.decipher(self.decipher_key, cipher_text)
        return plain_text

    def seek(self, stego_text):
        '''Seek the hidden cipher text from the given stego text by following
        the same forward sampling procedure.'''
        cipher_text = []
        remaining_bits = self.cipher_text_length
        for seq in stego_text:
            inds = [self.lm.vocabulary_index[token] for token in seq]
            cipher_text_fragment = self.recover_bits(inds, remaining_bits)
            cipher_text += cipher_text_fragment
            remaining_bits -= len(cipher_text_fragment)
        assert len(cipher_text) == self.cipher_text_length, 'Ciphertext must be of length %s.' % self.cipher_text_length
        return cipher_text

    def recover_bits(self, token_inds, remaining_bits):
        ind = self.lm.SOS_ind
        prefix = [ind]
        p = self.lm.p_next_token(prefix)
        cipher_text = []
        # Terminate the generation after we have consumed all indices or
        # have extracted all bits
        while 0 < len(token_inds) and 0 < remaining_bits:
            # Build Huffman codes for the conditional distribution
            heap = build_min_heap(p)
            hc = huffman_tree(heap)
            # Check if the total variation is low enough
            if tv_huffman(hc, p)[0] < self.tv_threshold:
                # We have controlled this step. Some bits are hidden.
                code = invert_code_tree(hc)
                # Look up the Huffman code for the token.
                ind = token_inds.pop(0)
                # Convert the Huffman code into bits
                # left => 0, right => 1
                cipher_text_fragment = [0 if bit == 'l' else 1 for bit in code[ind]]
                # Truncate possible trailing paddings
                cipher_text += cipher_text_fragment[:remaining_bits]
                remaining_bits -= len(cipher_text_fragment)
                print(remaining_bits)
                prefix += [ind]
                p = self.lm.p_next_token(prefix)
            else:
                # We did not control this step. Skip.
                prefix.append(token_inds.pop(0))
                p = self.lm.p_next_token(prefix)
        return cipher_text


if __name__ == '__main__':
    from gptlm import GptLanguageModel
    lm = GptLanguageModel()
    cipher_text_length = 32
    # tv_threshold = float('inf')
    tv_threshold = 0.08

    alice = Sender(lm, None, None, cipher_text_length, tv_threshold, seed=123)
    bob = Receiver(lm, None, None, cipher_text_length, tv_threshold)

    # sent_bits = list(np.random.choice(2, cipher_text_length))
    sent_bits = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]
    # sent_bits = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]
    print(sent_bits)
    stego_inds = alice.embed_bits(list(sent_bits))
    msg = lm.enc.decode(stego_inds)

    print(msg)

    token_inds = lm.enc.encode(msg)
    recovered_bits = bob.recover_bits(token_inds, cipher_text_length)
    print(recovered_bits)

    # Check
    print(recovered_bits == sent_bits)
    # stego_text = alice.hide(bits)
    # print(stego_text)
    # for seq in stego_text:
    #     print(''.join(seq))
