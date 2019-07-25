# lm-steganography

This is the codebase accompanying the publication [_Towards Near-imperceptible Steganographic Text_](https://arxiv.org/abs/1907.06679). It implements the design of linguistic steganographic system outlined in the paper, the `patient-Huffman` algorithm proposed, as well as the code we used for the empirical study in the paper.

## Preliminary

The steganographic systems we studied assume a cryptographic system that produces ciphertext to be encoded into stegotext. In this work, we encode the ciphertext into fluent stegotext by controlling the sampling from a language model. We focus on providing imperceptibility (steganographic secrecy) whereas the cryptographic security is provided by the cryptosystem.

## Usage

- [`example.ipynb`](https://github.com/falcondai/lm-steganography/blob/master/example.ipynb) contains a full example including the encryption/decryption steps.
- [`core.py`](https://github.com/falcondai/lm-steganography/blob/master/core.py) contains an illustrative minimal working example of the encoding/decoding of the stegosystem.
- You may need `GPT-2` (included as a git submodule) and the publicly released `GPT-2-117M` language model to generate stegotext.
- [This method](https://github.com/falcondai/lm-steganography/blob/master/core.py#L52) implements the `patient-Huffman` encoding algorithm. And its corresponding [decoding method](https://github.com/falcondai/lm-steganography/blob/master/core.py#L126).
- [This notebook](https://github.com/falcondai/lm-steganography/blob/master/entropy.ipynb) contains the plots and empirical analysis.

## Replication

Independent replications are more than welcome! Please bring them to our attention and we will list them here. For the original code that we used at the time of ACL submission, see the git commit tagged [`acl-2019`](https://github.com/falcondai/lm-steganography/tree/acl-2019).

## FAQ

This is intended as a research prototype. Please exercise caution when using it as a privacy protection tool.

- What is steganography?
  - Steganography is about hiding the fact that one is hiding something. It aims to avoid arousing suspicion in an eavesdropper (or a channel monitor) that some secretive communication is happening.
- What do we mean by imperceptibility?
  - Ideally, we want the steganographic communication to be imperceptible, that is, hiding in plain sight. In particular, we formalize this notion by asking how many samples it requires for the adversary to discover the presence of steganographic communication. This is precisely what total variation distance (and Kullback–Leibler divergence) between the effective language model and the base language model measures.
- Is there any known vulnerability?
  - In the paper's setting, we assume that the adversary is passive, i.e. it is merely observing the messages. One can imagine a stronger adversary that can also meddle with the messages themselves. The aim for such adversary may be to disrupt the steganographic communication without necessarily discovering its existence or disrupting the non-secretive communication. For example, by injecting common typos. This setting is sometimes called _robust_ steganography and the type of stegosystems we considered are **brittle** under such attack.

## Reference

Please cite our work if you find this repo or the associated paper useful.

```
Dai, Falcon Z and Cai, Zheng. Towards Near-imperceptible Steganographic Text. Proceedings of ACL. 2019.
```

```bibtex
@inproceedings{dai-cai-2019-towards,
    title = "Towards Near-imperceptible Steganographic Text",
    author = "Dai, Falcon Z and Cai, Zheng",
    booktitle = "Proceedings of Association for Computational Linguistics",
    month = july,
    year = "2019",
    publisher = "Association for Computational Linguistics"
  }
```
