import numpy as np

from huffman import build_min_heap, huffman_tree, tv_huffman
from lm import LanguageModel


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

    def __init__(self, lm, cipher_key, cipher, cipher_text_length, tv_threshold, seed=None):
        self.cipher_key = cipher_key
        self.cipher = cipher
        self.lm = lm
        self.tv_threshold = tv_threshold
        self.cipher_text_length = cipher_text_length
        self.random = np.random.RandomState(seed)
        self.acc_risk = 0

    def encrypt(self, plain_text):
        cipher_text = self.cipher(self.cipher_key, plain_text)
        return cipher_text

    def hide(self, cipher_text):
        '''We use the cipher text to control the forward sampling procedure
        in sampling tokens from the prefix language model.'''
        stego_text = []
        while len(cipher_text) > 0:
            inds = self.control_sample(cipher_text)
            # Look up the tokens
            stego_text += [self.lm.vocabulary[ind] for ind in inds]
        return stego_text

    def control_sample(self, coin_flips):
        '''We use a sequence of coin flips to control the generation of token
        indices from a language model. This returns _a sequence_ as defined by
        the model, e.g. sentence, paragraph.'''
        ind = self.lm.SOS_ind
        prefix = [ind]
        p = self.lm.p_next_token(prefix)
        # Terminate the generation after we generate the EOS token
        while ind != self.lm.EOS_ind:
            # There is still some cipher text to hide
            le = len(coin_flips)
            if le > 0:
                # Build Huffman codes for the conditional distribution
                # if 2 ** le < self.lm.vocabulary_size:
                #     # Truncate the distribution to the le-most likely tokens
                #     inds = np.argsort(p)[-le:]
                #     heap = build_min_heap(p, inds)
                # else:
                #     heap = build_min_heap(p)
                heap = build_min_heap(p)
                hc = huffman_tree(heap)
                # Check if the total variation is low enough
                if tv_huffman(hc, p) < self.tv_threshold:
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
        return prefix[:1]


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
        the same sampling procedure.'''


if __name__ == '__main__':
    lm = LanguageModel(['<s>', '</s>', 'a', 'b'], 0, 1)
    cipher_text_length = 128
    tv_threshold = 0.5
    alice = Sender(lm, None, None, cipher_text_length, tv_threshold, seed=123)
    stego_text = alice.hide([0, 1, 0, 1, 1, 1, 0, 0])
    print(stego_text)
