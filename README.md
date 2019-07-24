# lm-steganography

This is the codebase accompanying the publication [_Towards Near-imperceptible Steganographic Text_](https://arxiv.org/abs/1907.06679). It implements the design of linguistic steganographic system outlined in the paper, the `patient-Huffman` algorithm proposed, as well as the code we used for the empirical study in the paper.

## Framework

The steganographic systems we studied assume a cryptographic system that produces ciphertext to be encoded into stegotext. In this work, we encode the ciphertext into a fluent text by controlling the sampling from a language model. We focus on providing imperceptibility (steganographic secrecy) whereas the cryptographic security is provided by the cryptosystem, which is outside of the scope of this work.

## Usage

- [`example.ipynb`](https://github.com/falcondai/lm-steganography/blob/dai-dev/example.ipynb) contains a full example including the encryption/decryption steps.
- [`core.py`](https://github.com/falcondai/lm-steganography/blob/dai-dev/core.py) contains an illustrative minimal working example of the encoding/decoding of the stegosystem.
- You may need `GPT-2` (included as a git submodule) and the publicly released `GPT-2-117M` language model to generate stegotext.
- [This method](https://github.com/falcondai/lm-steganography/blob/master/core.py#L52) implements the `patient-Huffman` encoding algorithm. And its corresponding [decoding method](https://github.com/falcondai/lm-steganography/blob/master/core.py#L126).
- [This notebook](https://github.com/falcondai/lm-steganography/blob/master/entropy.ipynb) contains the plots and empirical analysis.

## Replication

Independent replications are more than welcome! Please bring them to our attention and we will list them here. For the original code that we used at the time of ACL submission, see the git commit tagged [`acl-2019`](https://github.com/falcondai/lm-steganography/tree/acl-2019).

## Reference

Please cite our work if you find this repo or the associated paper useful.

```
Dai, Falcon Z and Cai, Zheng. Towards Near-imperceptible Steganographic Text. Proceedings of ACL 2019.
```

```bibtex
@inproceedings{dai-cai-2019-towards,
    title = "Towards Near-imperceptible Steganographic Text",
    author = "Dai, Falcon Z and
      Cai, Zheng",
    booktitle = "Proceedings of Association for Computational Linguistics",
    month = july,
    year = "2019",
    publisher = "Association for Computational Linguistics"
  }
```
