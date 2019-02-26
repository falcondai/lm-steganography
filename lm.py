class LanguageModel:
    '''
    Abstraction for prefix language models
    p(sentence) = p(next token|prefix) p(prefix)
    '''

    def __init__(self, vocabulary, SOS_ind, EOS_ind):
        self.SOS_ind = SOS_ind
        self.EOS_ind = EOS_ind
        # Mapping from indices to tokens
        self.vocabulary = vocabulary
        # Inverse map from tokens to indices
        self.vocabulary_index = lambda token: self.vocabulary.index(token)
        self.vocabulary_size = len(self.vocabulary)

    def p_next_token(self, prefix):
        '''Returns the distribution over the next token given the prefix
        represented by a list of indices.'''
        raise NotImplementedError()

    def perplexity(self, sentence):
        '''Returns -log p(sentence)'''
        raise NotImplementedError()


class NgramLanguageModel(LanguageModel):
    def __init__(self, vocabulary, SOS_ind, EOS_ind):
        super().__init__(vocabulary, SOS_ind, EOS_ind)

    def p_next_token(self, prefix):
        pass


class StatefulLanguageModel(LanguageModel):
    def __init__(self, vocabulary, SOS_ind, EOS_ind):
        self.prefix_to_state = {}


class GptLanguageModel(StatefulLanguageModel):
    '''GPT-2 as a language model.'''
    pass


if __name__ == '__main__':
    pass
