import json, random, time
from math import gcd
import numpy as np
from trilayer_cipher import (ALPHA, clean, hill_matrix, encrypt, decrypt, shift_enc, shift_dec)
from attack import (load_bigram_model, frequency_attack, known_plaintext_attack, to_arr, FREQ)

SEED = 2026
rng = random.Random(SEED)
text = clean(open("corpus.txt", errors="ignore").read())
cut = int(len(text) * 0.7)
BG = load_bigram_model(text[:cut])          
TEST = text[cut:]                           
UNITS = [x for x in range(1, 26) if gcd(x, 26) == 1]


def random_key(L=None):
    L = L or rng.randint(10, 16)
    while True:
        h = "".join(rng.choice(ALPHA) for _ in range(4))
        try:
            hill_matrix(h); break
        except ValueError:
            pass
    return dict(a=rng.choice(UNITS), b=rng.randrange(26),
                vig_key="".join(rng.choice(ALPHA) for _ in range(L)), hill_key=h)


def msg(n):
    s = rng.randrange(len(TEST) - n)
    return TEST[s:s + n]


def ic(t):
    c = np.bincount(to_arr(t), minlength=26)
    return float((c * (c - 1)).sum() / (len(t) * (len(t) - 1)))


res = {}
ok = 0
for _ in range(300):
    k = random_key(); p = msg(rng.randint(1, 200)); p += "X" if len(p) % 2 else ""
    ok += decrypt(encrypt(p, **k), **k) == p
res["roundtrip"] = f"{ok}/300"
print("round-trip tests:", res["roundtrip"])

p = msg(4000); k = random_key(); c = encrypt(p, **k)
res["ic"] = {"english_theory": float((FREQ ** 2).sum()), "plaintext": ic(p), "ciphertext": ic(c), "random_26": 1 / 26}
res["freq_plain"] = (np.bincount(to_arr(p), minlength=26) / len(p)).tolist()
res["freq_cipher"] = (np.bincount(to_arr(c), minlength=26) / len(c)).tolist()
print("IC:", {a: round(b, 4) for a, b in res["ic"].items()})

print("\nATTACK A (ciphertext-only frequency analysis), 100 random keys per length")
resA = []
for n in (200, 400, 600, 800, 1000, 1500, 2000, 4000):
    T, full, accs, times = 100, 0, [], []
    for _ in range(T):
        k = random_key(); pt = msg(n); ct = encrypt(pt, **k); pt = pt + ("X" if len(pt) % 2 else "")
        rec, P, info = frequency_attack(ct, BG)
        acc = sum(a == b for a, b in zip(rec, pt)) / len(pt) if rec else 0.0
        full += acc == 1.0; accs.append(acc); times.append(info["time"])
    resA.append(dict(n=n, full=full / T, acc=float(np.mean(accs)), t=float(np.mean(times))))
    print(f"  n={n:5d}  full recovery {full / T:5.0%}   mean letters correct {np.mean(accs):6.1%}   {np.mean(times):.3f}s")
res["attackA"] = resA

print("\nATTACK B (known plaintext), 200 random keys per crib length, ciphertext 1000 letters")
resB = []
for m in (10, 16, 20, 24, 28, 32, 36, 40, 50, 60):
    T, full, times = 200, 0, []
    for _ in range(T):
        k = random_key(); pt = msg(1000); ct = encrypt(pt, **k)
        rec, P, info = known_plaintext_attack(ct, pt[:m])
        full += rec == pt; times.append(info["time"])
    resB.append(dict(m=m, full=full / T, t=float(np.mean(times))))
    print(f"  crib={m:3d} letters  full recovery {full / T:5.0%}   {np.mean(times) * 1000:.1f} ms")
res["attackB"] = resB

perL = []
for L in range(10, 17):
    P = L // gcd(L, 2)
    row = {"L": L, "P": P, "formula": 2 * (P + 2)}
    for m in range(8, 90, 2):
        okk = 0
        for _ in range(40):
            k = random_key(L); pt = msg(600); ct = encrypt(pt, **k)
            okk += known_plaintext_attack(ct, pt[:m])[0] == pt
        if okk == 40:
            row["min_crib_all_ok"] = m; break
    perL.append(row)
    print("  ", row)
res["perL"] = perL

print("\nTIMING (pure Python, best of 3), seconds")
timing = []
for n in (1_000, 10_000, 100_000, 1_000_000):
    pt = (TEST * (n // len(TEST) + 1))[:n]; k = dict(a=7, b=11, vig_key="NETWORKSECURITY", hill_key="HILL")
    def best(f, *a):
        out = []
        for _ in range(3):
            t0 = time.perf_counter(); r = f(*a); out.append(time.perf_counter() - t0)
        return min(out), r
    te, ct = best(encrypt, pt, **k) if False else best(lambda: encrypt(pt, **k))
    td, _ = best(lambda: decrypt(ct, **k))
    se, sct = best(lambda: shift_enc(pt, 3)); sd, _ = best(lambda: shift_dec(sct, 3))
    timing.append(dict(n=n, tri_enc=te, tri_dec=td, shift_enc=se, shift_dec=sd))
    print(f"  n={n:8d}  TriLayer enc {te:.4f} dec {td:.4f} | Shift enc {se:.4f} dec {sd:.4f} | ratio {te / se:.1f}x")
res["timing"] = timing

pt = msg(1000); sct = shift_enc(pt, 17)
t0 = time.perf_counter()
best_k = max(range(26), key=lambda k: float(np.log(FREQ)[to_arr(shift_dec(sct, k))].sum()))
res["shift_break_ms"] = (time.perf_counter() - t0) * 1000
print(f"\nShift cipher broken by frequency analysis in {res['shift_break_ms']:.1f} ms (key found: {best_k})")

json.dump(res, open("results.json", "w"), indent=1)
