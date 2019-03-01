import numpy as np
from numpy.random import choice
from gptlm import GptLanguageModel


def saving(path, data):
    with open(path, 'wb') as f:
       np.save(f,data)

if __name__ == '__main__':
    # Example
    lm = GptLanguageModel()
    # with open('./ptb.valid.txt', 'r') as f:
    #     files = f.read()
    N_rounds = 50
    T_length = 40
    index_set = []
    sample_set = []
    logits_set = []
    n_vocab = lm.vocabulary_size
    for i in range(N_rounds):
        prefix = [lm.SOS]
        
        logits_seq = np.zeros([T_length, n_vocab])
        for t in range(T_length):
            logits = lm.p_next_token(prefix)
            next_token_sample = choice(n_vocab, p=logits)
            prefix.append(next_token_sample)
            logits_seq[t] = np.array(logits)
        
        index_set.append(prefix)
        sample_set.append([lm.enc.decoder[i] for i in prefix])
        logits_set.append(logits_seq)

    saving("./samples/sample_sents_idxs.npy", np.asarray(index_set))
    saving("./samples/sample_sents_tokens.npy", np.asarray(sample_set))
    saving("./samples/sample_sents_prob.npy", np.asarray(logits_set))