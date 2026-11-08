# TriLayer Cipher (Affine + Vigenère + Hill) and its cryptanalysis

Network and Information Security (CT-486), Complex Computing Problem (CCP).

## Files
|File                 | Purpose                                                                                     |

`trilayer_cipher.py` | Encryption / decryption (Affine -> Vigenère -> Hill 2x2) and a Shift-cipher baseline        |

`attack.py`          | Attack A (ciphertext-only frequency analysis) and Attack B (known plaintext)                |

`benchmark.py`       | Random-key success rates, timings, Shift comparison -> `results.json`                       |

`make_figures.py`    | Draws the report figures from `results.json`                                                |

`corpus.txt`         | English text (first 70 % trains the attacker's bigram model, last 30 % gives test messages) |

## Requirements
Python 3.9+ and `numpy`. `matplotlib` is only needed for `make_figures.py`.

## Run
```
python3 trilayer_cipher.py   # encrypt/decrypt demo
python3 attack.py            # both attacks on a 1,150-letter demo message
python3 benchmark.py         # full experiments, about 2-3 minutes, seed 2026
python3 make_figures.py      # figures from results.json
```

## Key format
`encrypt(plaintext, a, b, vig_key, hill_key)`: `a` coprime with 26, `b` in 0..25, `vig_key` at least 10 letters,
`hill_key` 4 letters (2x2 matrix, determinant coprime with 26, e.g. `HILL`).
Input is upper-cased and reduced to A-Z; an odd length gets one `X` appended.
