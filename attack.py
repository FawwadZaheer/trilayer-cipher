import time
from itertools import combinations, permutations
from math import gcd

import numpy as np

from trilayer_cipher import M, clean, mat_inv

ENGLISH_PCT = [8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772,
               4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978,
               2.360, 0.150, 1.974, 0.074]
FREQ = np.array(ENGLISH_PCT) / sum(ENGLISH_PCT)
LOGF = np.log(FREQ)
SL = np.array([[LOGF[(a - w) % 26] for w in range(26)] for a in range(26)])


def to_arr(t):
    return np.frombuffer(t.encode(), dtype=np.uint8).astype(np.int64) - 65


def to_str(a):
    return (np.asarray(a, dtype=np.int64) % 26 + 65).astype(np.uint8).tobytes().decode()


def load_bigram_model(corpus_text):
    """English bigram log-probabilities learned from a training text (used only to order 2 rows)."""
    a = to_arr(clean(corpus_text))
    cnt = np.bincount(a[:-1] * 26 + a[1:], minlength=676).astype(float) + 1.0
    return np.log(cnt / cnt.sum()).reshape(26, 26)


def bigram_score(text_arr, bg):
    return bg[text_arr[:-1], text_arr[1:]].sum()


PRIMITIVE_ROWS = [(n0, n1) for n0 in range(26) for n1 in range(26)
                  if gcd(gcd(n0, n1), M) == 1]          


def _hdr(t):
    print("\n" + "-" * 74 + f"\n{t}\n" + "-" * 74)


def _letters(v):
    return "".join(chr(65 + int(x) % 26) for x in v)


def frequency_attack(ct, bigram, p_range=range(5, 25), n_top=4, eps=0.02, verbose=False):
    """Ciphertext-only attack. Returns (plaintext, period_in_blocks, info). verbose=True prints every step."""
    t0 = time.time()
    ct = clean(ct)
    assert len(ct) % 2 == 0, "ciphertext length must be even"
    C = to_arr(ct)
    C0, C1 = C[0::2], C[1::2]
    B = len(C0)
    j = np.arange(B)
    P_list = [P for P in p_range if P * 4 <= B]            
    if verbose:
        _hdr("ATTACK A, STEP 1: read the ciphertext and cut it into 2-letter blocks")
        print(f"    {len(ct)} letters -> {B} blocks.   first 6 blocks: "
              + "  ".join(f"({ct[2*i]},{ct[2*i+1]})={int(C0[i])},{int(C1[i])}" for i in range(6)))
        print("    Model: every block is C_j = M' * P_j + u_(j mod P)   (Affine+Vigenere+Hill = ONE affine map).")
        print("    Unknown: matrix M' and key period P. Idea: one ROW of M'^-1 applied to all blocks gives a")
        print("    stream = (plain English letters) + (periodic Vigenere shift), which frequency analysis can solve.")
    if not P_list:
        return None, None, {"time": time.time() - t0}
    colidx = {P: (j % P) * 26 for P in P_list}

    if verbose:
        _hdr("STEP 2: try every row (n0, n1) and every period P")
        print(f"    {len(PRIMITIVE_ROWS)} primitive rows (gcd(n0,n1,26)=1) x {len(P_list)} periods "
              f"({P_list[0]}..{P_list[-1]}) = {len(PRIMITIVE_ROWS) * len(P_list)} fits.")
        print("    One fit = x_j = (n0*C0_j + n1*C1_j) mod 26, split x into P columns (j mod P),")
        print("    pick for each column the shift with the best English log-likelihood, add the columns up.")
    scores = np.zeros((len(PRIMITIVE_ROWS), len(P_list)))
    for ri, (n0, n1) in enumerate(PRIMITIVE_ROWS):
        x = (n0 * C0 + n1 * C1) % 26
        for pi, P in enumerate(P_list):
            counts = np.bincount(colidx[P] + x, minlength=P * 26).reshape(P, 26)
            scores[ri, pi] = (counts @ SL).max(axis=1).sum() / B  

    top2 = np.sort(scores, axis=0)[-2:, :].sum(axis=0)
    P_star = next(P_list[i] for i in range(len(P_list)) if top2[i] >= top2.max() - eps)
    pi = P_list.index(P_star)
    if verbose:
        _hdr("STEP 3: which period fits English best? (sum of the two best rows, higher = better)")
        print("    period P | best row fit | 2nd best row | total  ")
        srt = np.sort(scores, axis=0)
        for k, P in enumerate(P_list):
            mark = "   <== chosen" if P == P_star else ""
            print(f"    {P:>8} | {srt[-1, k]:>12.3f} | {srt[-2, k]:>12.3f} | {top2[k]:>7.3f}{mark}")
        print(f"    Rule: take the SMALLEST period whose total is within {eps} of the maximum "
              f"({top2.max():.3f}) -> P* = {P_star} blocks.")
        print(f"    => key length L = {P_star} (odd) or {2 * P_star} (even). Random text would score about -6.5 to -7.")

    cand_rows = np.argsort(scores[:, pi])[::-1][:n_top]
    if verbose:
        _hdr(f"STEP 4: best rows at P* = {P_star}")
        for r in cand_rows:
            print(f"    row {PRIMITIVE_ROWS[r]}  fit = {scores[r, pi]:.3f}")
        print("    The two true rows of M'^-1 stand far above the rest. Which is row 1 and which row 2")
        print("    is unknown, so both orders are decoded and judged with English bigram statistics.")

    def solve_stream(ri, P):
        n0, n1 = PRIMITIVE_ROWS[ri]
        x = (n0 * C0 + n1 * C1) % 26
        counts = np.bincount(colidx.get(P, (j % P) * 26) + x, minlength=P * 26).reshape(P, 26)
        w = (counts @ SL).argmax(axis=1)                 
        return (x - w[j % P]) % 26, w

    best = None
    if verbose:
        _hdr("STEP 5: try the row orders")
    for ra, rb in permutations(cand_rows, 2):
        (a0, a1), (b0, b1) = PRIMITIVE_ROWS[ra], PRIMITIVE_ROWS[rb]
        det = (a0 * b1 - a1 * b0) % 26
        if gcd(det, 26) != 1:
            continue                                       
        pe, we = solve_stream(ra, P_star)
        po, wo = solve_stream(rb, P_star)
        txt = np.empty(2 * B, dtype=np.int64)
        txt[0::2], txt[1::2] = pe, po
        sc = bigram_score(txt, bigram)
        if verbose:
            print(f"    rows ({PRIMITIVE_ROWS[ra]}, {PRIMITIVE_ROWS[rb]})  det={det}  "
                  f"bigram score {sc:9.1f}   text starts {_letters(txt[:24])}")
        if best is None or sc > best[0]:
            best = (sc, txt, ra, rb, we, wo)
    if best is None:
        return None, P_star, {"time": time.time() - t0}
    sc, txt, ra, rb, we, wo = best
    N = [list(PRIMITIVE_ROWS[ra]), list(PRIMITIVE_ROWS[rb])]
    info = {"time": time.time() - t0, "period_blocks": P_star,
            "key_period_letters": f"{2 * P_star} (even L) or {P_star} (odd L)",
            "column_fits": len(PRIMITIVE_ROWS) * len(P_list)}
    if verbose:
        _hdr("STEP 6: recovered key material")
        print(f"    N = M'^-1 = {N[0]} / {N[1]}        (best bigram score {sc:.1f})")
        print(f"    M' = N^-1 = {mat_inv(N)[0]} / {mat_inv(N)[1]}   (this is a*H of the real key)")
        print(f"    Vigenere column shifts, even stream: {_letters(we)}   odd stream: {_letters(wo)}")
        print("    (the shifts are the real Vigenere key only up to the Hill/Affine mixing - they are all that is")
        print("     needed to decrypt, which is why the 21-character key gives no extra protection.)")
        _hdr("STEP 7: decrypt every block with these shifts")
        print(f"    time {info['time']:.2f}s, {info['column_fits']} fits")
        print("    plaintext start:", _letters(txt[:80]), "...")
    return to_str(best[1]), P_star, info

def _det(m):
    return (m[0][0] * m[1][1] - m[0][1] * m[1][0]) % M


def known_plaintext_attack(ct, crib, p_range=range(5, 31), verbose=False):
    """Known-plaintext attack; `crib` = the first letters of the message. Returns (plaintext, P, info)."""
    t0 = time.time()
    ct, crib = clean(ct), clean(crib)
    assert len(ct) % 2 == 0
    C = to_arr(ct).reshape(-1, 2)
    mb = len(crib) // 2
    Pk = to_arr(crib[:2 * mb]).reshape(-1, 2)
    B = len(C)
    if verbose:
        _hdr("ATTACK B, STEP 1: known plaintext and its ciphertext blocks")
        print(f"    crib = {crib}  ({len(crib)} letters = {mb} blocks)")
        print("    block j: plaintext (p1,p2)  ->  ciphertext (c1,c2)")
        for i in range(min(6, mb)):
            print(f"    {i:>3}: {tuple(int(v) for v in Pk[i])} ({_letters(Pk[i])})  ->  {tuple(int(v) for v in C[i])} ({_letters(C[i])})")
        print("    Model: C_j = M' P_j + u_(j mod P). Blocks P apart share u, so")
        print("           C_(j+P) - C_j = M' (P_(j+P) - P_j)   -> u cancels, only M' is left.")
        _hdr("STEP 2: try each key period P")
    cands = []
    for P in p_range:
        nd = mb - P
        if nd < 2 or P > B:
            if verbose and P <= 16:
                print(f"    P = {P:>2}: only {max(nd, 0)} difference(s) available, need 2 -> skip")
            continue
        dc = (C[P:mb] - C[:nd]) % 26
        dp = (Pk[P:mb] - Pk[:nd]) % 26
        Mp = None
        for i1, i2 in combinations(range(nd), 2):
            Dp = [[int(dp[i1][0]), int(dp[i2][0])], [int(dp[i1][1]), int(dp[i2][1])]]
            if gcd(_det(Dp), 26) != 1:
                continue
            Dc = [[int(dc[i1][0]), int(dc[i2][0])], [int(dc[i1][1]), int(dc[i2][1])]]
            Dpi = mat_inv(Dp)
            cand = [[sum(Dc[r][k] * Dpi[k][c] for k in range(2)) % 26 for c in range(2)] for r in range(2)]
            if gcd(_det(cand), 26) != 1:
                continue
            ca = np.array(cand)
            if np.all((dp @ ca.T) % 26 == dc):            
                Mp = ca
                if verbose:
                    print(f"    P = {P:>2}: {nd} differences; using differences #{i1},#{i2}:")
                    print(f"           Dp (plain diffs) = {Dp[0]} / {Dp[1]}   Dc (cipher diffs) = {Dc[0]} / {Dc[1]}")
                    print(f"           M' = Dc * Dp^-1 = {cand[0]} / {cand[1]}   -> explains all {nd} differences: YES")
                break
        if Mp is None:
            if verbose:
                print(f"    P = {P:>2}: {nd} differences, no consistent matrix -> wrong period")
            continue
        Minv = np.array(mat_inv(Mp.tolist()))
        U = np.array([(C[s] - Mp @ Pk[s]) % 26 for s in range(P)])
        plain = ((C - U[np.arange(B) % P]) @ Minv.T) % 26
        flat = plain.reshape(-1)
        if not np.array_equal(plain[:mb], Pk):
            continue
        fit = float(LOGF[flat].sum())
        cands.append((fit, -P, P, flat, Mp, U))
        if verbose:
            print(f"           u vectors (one per class): {[tuple(int(v) for v in u) for u in U]}")
            print(f"           decrypts the crib correctly; English fit of whole text = {fit:.1f}")
    if not cands:
        if verbose:
            print("    No period gave a consistent solution: the known plaintext is too short.")
        return None, None, {"time": time.time() - t0}
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    fit, _, P, flat, Mp, U = cands[0]
    if verbose:
        _hdr("STEP 3: choose the best candidate and decrypt everything")
        print(f"    best period P = {P} blocks (key length L = {P} odd or {2 * P} even), English fit {fit:.1f}")
        print(f"    M' = {Mp.tolist()}   time {1000 * (time.time() - t0):.1f} ms")
        print("    plaintext start:", _letters(flat[:80]), "...")
    return to_str(flat), P, {"time": time.time() - t0, "period_blocks": P}


SAMPLE = """Network security protects the confidentiality integrity and availability of information as it
moves between computers. A firewall inspects incoming and outgoing traffic and blocks connections that
break the rules set by the administrator. Encryption turns readable data into unreadable ciphertext so
that an attacker who captures packets on the wire learns nothing useful. Classical ciphers such as the
Caesar shift and the Playfair square were once considered strong but modern computers can break them in
seconds. The lesson for every security engineer is that a cipher must never rely on secrecy of the
algorithm alone and that combining weak layers does not automatically create a strong system. Statistical
patterns in language survive substitution and transposition and these patterns give the cryptanalyst a
way into the message. A linear layer followed by another linear layer is still a linear layer, so the
whole construction can be written as a single matrix equation and solved with a little algebra. Good
ciphers therefore insert a non linear step between the linear ones so that no such shortcut exists. An
administrator who understands this will prefer a vetted standard such as AES over any home made design
because years of public analysis have not found a practical weakness in it while home made schemes
usually fall to a student with a laptop in an afternoon."""

if __name__ == "__main__":
    from trilayer_cipher import encrypt
    text = open("corpus.txt", errors="ignore").read()
    bg = load_bigram_model(text[: int(len(text) * 0.7)])
    KEY = dict(a=7, b=11, vig_key="NETWORKSECURITY", hill_key="HILL")
    ct = encrypt(SAMPLE, **KEY)
    pt = clean(SAMPLE)
    pt += "X" if len(pt) % 2 else ""
    print("=" * 74)
    print("SETUP: the attacker only sees the ciphertext (and, for attack B, a few known letters)")
    print("=" * 74)
    print(f"    secret key (hidden from the attacker): {KEY}")
    print(f"    ciphertext: {len(ct)} letters, starts {ct[:60]}...")

    print("\n" + "#" * 74 + "\n# ATTACK A - ciphertext-only frequency analysis\n" + "#" * 74)
    rec, P, info = frequency_attack(ct, bg, verbose=True)
    acc = sum(x == y for x, y in zip(rec, pt)) / len(pt)
    print(f"\n    RESULT A: period P = {P} blocks (true key length 15 -> P = 15), "
          f"letters correct {acc:.1%}, time {info['time']:.2f}s")

    print("\n" + "#" * 74 + "\n# ATTACK B - known plaintext (first 40 letters known)\n" + "#" * 74)
    rec, P, info = known_plaintext_attack(ct, pt[:40], verbose=True)
    acc = sum(x == y for x, y in zip(rec, pt)) / len(pt)
    print(f"\n    RESULT B: period P = {P} blocks, letters correct {acc:.1%}, time {info['time'] * 1000:.1f} ms")
