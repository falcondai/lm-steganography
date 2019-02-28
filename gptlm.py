#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf

import encoder
from lm import LanguageModel

# import model, sample, encoder
import importlib.util
spec_model = importlib.util.spec_from_file_location("module.model", "./external/gpt-2/src/model.py")
model = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(model)


class GptLanguageModel(LanguageModel):
    def __init__(self, model_name='117M', seed=None, nsamples=1, batch_size=None, length=None, tempterature=1, top_k=0):
        if batch_size is None:
           self.batch_size = 1

        self.nsamples = nsamples
        base_path = "./external/gpt-2/"
        self.enc = encoder.get_encoder_custom(base_path, model_name)
        self.hparams = model.default_hparams()
        with open(os.path.join(base_path, 'models', model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))
        self.SOS = self.enc.encoder['<|endoftext|>']
        if length is None:
            self.length = self.hparams.n_ctx // 2
        elif length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

        self.sess = tf.Session()
        self.context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.lm_output = model.model(hparams=self.hparams, X=self.context[:, :], past=None, reuse=tf.AUTO_REUSE)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(base_path, 'models', model_name))
        saver.restore(self.sess, ckpt)

        self.EOS_ind = self.SOS
        self.SOS_ind = self.SOS
        self.vocabulary_index = self.enc.encoder
        self.vocabulary = self.enc.decoder
        self.vocabulary_size = self.hparams.n_vocab

    def p_next_token(self, prefix):
        # raw_text = prefix
        # if not raw_text:
        #     print('Prompt should not be empty!')
        #     raise ValueError("must have prefix tokens.")
        context_tokens = prefix
        # print('prefix', context_tokens)
        context_tk_reshape = np.asarray(context_tokens).reshape((self.batch_size, -1))

        out = self.sess.run(self.lm_output, feed_dict={
                self.context: context_tk_reshape})
        logits = out['logits'][0, -1]
        max_logit = logits.max()
        p = np.exp(logits - max_logit)
        p /= p.sum()
        return p

    def perplexity(self, sentence):
        sos_padding = np.array([self.SOS for i in range(self.batch_size)]).reshape((self.batch_size, -1))
        sent_tokens = self.enc.encode(sentence)
        sent_reshape = np.asarray(sent_tokens).reshape((self.batch_size, -1))
        context = np.concatenate([sos_padding, sent_reshape], axis = 1)

        out = self.sess.run(self.lm_output, feed_dict={self.context: context})
        # logits should have shape [batch_size, length]
        ppl = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = sent_reshape.reshape((-1, self.batch_size)),
                logits = out['logits']
            )
        return ppl


if __name__ == '__main__':
    def entropy(p):
        return -np.sum(p * np.log2(p))

    # Example
    lm = GptLanguageModel()
    prefix = [lm.SOS]
    logits = lm.p_next_token(prefix)
    print(logits)
    inds = logits.argsort()[-10:]
    print([lm.enc.decoder[i] for i in inds[::-1]])
    print(entropy(logits))

    # Low entropy for some prefixes
    prefix = 'I have a lot'
    i_have_a_lot = lm.enc.encode(prefix)
    logits = lm.p_next_token(i_have_a_lot)
    print(prefix)
    print(logits)
    inds = logits.argsort()[-10:]
    print([lm.enc.decoder[i] for i in inds[::-1]])
    print(entropy(logits))

    # High entropy for some other prefixes
    prefix = 'I like your'
    i_like_your = lm.enc.encode(prefix)
    logits = lm.p_next_token(i_like_your)
    print(prefix)
    print(logits)
    inds = logits.argsort()[-10:]
    print([lm.enc.decoder[i] for i in inds[::-1]])
    print(entropy(logits))
