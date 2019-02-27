#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

# import model, sample, encoder
import importlib.util
spec_model = importlib.util.spec_from_file_location("module.model", "./external/gpt-2/src/model.py")
model = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(model)
import encoder
from core import LanguageModel


class GptLanguageModel(LanguageModel):
    def __init__(self, model_name='117M', seed=None, nsamples=1, batch_size=None, length=None, tempterature=1, top_k=0):
        if batch_size is None:
           self.batch_size = 1

        self.nsamples = nsamples
        base_path = "./external/gpt-2/"
        self.enc = encoder.get_encoder_custom(model_name)
        self.hparams = model.default_hparams()
        with open(os.path.join(base_path, 'models', model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))
        self.SOS = self.enc.encoder['<|endoftext|>']
        if length is None:
            self.length = self.hparams.n_ctx // 2
        elif length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        self.sess = tf.Session()
        self.context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        self.lm_output = model.model(hparams=self.hparams, X=self.context[:, :-1], past=None, reuse=tf.AUTO_REUSE)
        
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(base_path, 'models', model_name))
        saver.restore(self.sess, ckpt)
    
    def p_next_token(self, prefix):

        raw_text = prefix
        if not raw_text:
            print('Prompt should not be empty!')
            raise ValueError("must have prefix tokens.") 
        context_tokens = self.enc.encode(raw_text)
        print(context_tokens)
        context_tk_reshape = np.asarray(context_tokens).reshape((self.batch_size, -1)) 

        out = self.sess.run(self.lm_output, feed_dict={
                self.context: context_tk_reshape})
        p_next_tk = out['logits']
        return p_next_tk

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
