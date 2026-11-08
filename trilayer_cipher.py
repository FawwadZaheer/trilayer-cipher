from math import gcd

M = 26
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_VIG_LEN = 10

def clean(text: str) -> str:
    """Upper-case and keep A-Z only."""
    return "".join(c for c in text.upper() if "A" <= c <= "Z")


def pad(text: str) -> str:
    return text + "X" if len(text) % 2 else text


def modinv(a: int, m: int = M) -> int:
    a %= m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError(f"{a} has no inverse mod {m}")


def hill_matrix(key: str):
    k = [ALPHA.index(c) for c in clean(key)]
    if len(k) != 4:
        raise ValueError("hill_key must be exactly 4 letters")
    det = (k[0] * k[3] - k[1] * k[2]) % M
    if gcd(det, M) != 1:
        raise ValueError(f"Hill matrix not invertible mod 26 (det = {det})")
    return [[k[0], k[1]], [k[2], k[3]]]


def mat_inv(m):
    det = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) % M
    di = modinv(det)
    return [[(m[1][1] * di) % M, (-m[0][1] * di) % M],
            [(-m[1][0] * di) % M, (m[0][0] * di) % M]]


def check_key(a, b, vig_key, hill_key):
    if gcd(a, M) != 1:
        raise ValueError("a must be coprime with 26")
    if not 0 <= b < M:
        raise ValueError("b must be in 0..25")
    if len(clean(vig_key)) < MIN_VIG_LEN:
        raise ValueError(f"vig_key needs at least {MIN_VIG_LEN} letters")
    hill_matrix(hill_key)

def affine_enc(t, a, b):
    return "".join(chr(65 + (a * (ord(c) - 65) + b) % M) for c in t)


def affine_dec(t, a, b):
    ai = modinv(a)
    return "".join(chr(65 + (ai * (ord(c) - 65 - b)) % M) for c in t)

def vig_enc(t, key):
    k = [ord(c) - 65 for c in clean(key)]
    L = len(k)
    return "".join(chr(65 + (ord(c) - 65 + k[i % L]) % M) for i, c in enumerate(t))


def vig_dec(t, key):
    k = [ord(c) - 65 for c in clean(key)]
    L = len(k)
    return "".join(chr(65 + (ord(c) - 65 - k[i % L]) % M) for i, c in enumerate(t))

def hill_apply(t, m):
    out = []
    for i in range(0, len(t), 2):
        x, y = ord(t[i]) - 65, ord(t[i + 1]) - 65
        out.append(ALPHA[(m[0][0] * x + m[0][1] * y) % M])
        out.append(ALPHA[(m[1][0] * x + m[1][1] * y) % M])
    return "".join(out)

def encrypt(plaintext, a, b, vig_key, hill_key):
    check_key(a, b, vig_key, hill_key)
    t = pad(clean(plaintext))
    t = affine_enc(t, a, b)
    t = vig_enc(t, vig_key)
    return hill_apply(t, hill_matrix(hill_key))


def decrypt(ciphertext, a, b, vig_key, hill_key):
    check_key(a, b, vig_key, hill_key)
    t = hill_apply(clean(ciphertext), mat_inv(hill_matrix(hill_key)))
    t = vig_dec(t, vig_key)
    return affine_dec(t, a, b)

def shift_enc(t, k):
    return "".join(chr(65 + (ord(c) - 65 + k) % M) for c in t)


def shift_dec(t, k):
    return shift_enc(t, -k)

def _wrap(label, text, width=64):
    print(f"    {label}")
    for i in range(0, len(text), width):
        print("      " + text[i:i + width])


def _line(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def encrypt_verbose(plaintext, a, b, vig_key, hill_key, show=8):
    """Same result as encrypt(), but prints every step. `show` = how many letters/blocks to detail."""
    _line("ENCRYPTION - STEP 0: check the key")
    check_key(a, b, vig_key, hill_key)
    H = hill_matrix(hill_key)
    det = (H[0][0] * H[1][1] - H[0][1] * H[1][0]) % M
    K = clean(vig_key)
    print(f"    Affine   a = {a}  (gcd({a}, 26) = {gcd(a, M)} -> OK),  b = {b}")
    print(f"    Vigenere key = {K}  ({len(K)} letters, minimum is {MIN_VIG_LEN} -> OK)")
    print(f"    Hill key word = {clean(hill_key)}  -> numbers {[ord(c) - 65 for c in clean(hill_key)]}")
    print(f"    Hill matrix H = {H[0]} / {H[1]}")
    print(f"    det(H) = {H[0][0]}*{H[1][1]} - {H[0][1]}*{H[1][0]} = {det} (mod 26),  gcd({det}, 26) = {gcd(det, M)} -> invertible")

    _line("STEP 1: prepare the plaintext (keep A-Z only, pad to even length)")
    c1 = clean(plaintext)
    t = pad(c1)
    print(f"    original  : {plaintext}")
    _wrap(f"cleaned   : {len(c1)} letters", c1)
    print("    padding   : " + ("odd length -> appended 'X'" if len(t) != len(c1) else "length already even, no padding"))
    _wrap(f"text to encrypt ({len(t)} letters):", t)

    _line("STEP 2 - LAYER 1, AFFINE:  y = (a*x + b) mod 26")
    x1 = affine_enc(t, a, b)
    print(f"    {'letter':>6} {'x':>3}   a*x+b          mod 26   result")
    for ch in t[:show]:
        x = ord(ch) - 65
        v = a * x + b
        print(f"    {ch:>6} {x:>3}   {a}*{x}+{b} = {v:<5}  {v % M:>5}   {chr(65 + v % M)}")
    print("    ...")
    _wrap("after Affine:", x1)

    _line("STEP 3 - LAYER 2, VIGENERE:  y = (x + key[i mod len]) mod 26")
    x2 = vig_enc(x1, vig_key)
    kk = "".join(K[i % len(K)] for i in range(min(len(x1), 60)))
    print("    key stream repeated under the text (first 60 letters):")
    print(f"      text : {x1[:60]}")
    print(f"      key  : {kk}")
    print(f"    {'pos':>3} {'text':>4} {'key':>4}   sum    mod 26   result")
    for i in range(min(show, len(x1))):
        xv, kv = ord(x1[i]) - 65, ord(K[i % len(K)]) - 65
        print(f"    {i:>3} {x1[i]:>4} {K[i % len(K)]:>4}   {xv}+{kv} = {xv + kv:<3}  {(xv + kv) % M:>5}   {x2[i]}")
    print("    ...")
    _wrap("after Vigenere:", x2)

    _line("STEP 4 - LAYER 3, HILL 2x2:  (c1, c2) = H * (y1, y2) mod 26")
    c = hill_apply(x2, H)
    print(f"    H = {H[0]} / {H[1]}")
    for j in range(min(show // 2, len(x2) // 2)):
        y1, y2 = ord(x2[2 * j]) - 65, ord(x2[2 * j + 1]) - 65
        r1 = H[0][0] * y1 + H[0][1] * y2
        r2 = H[1][0] * y1 + H[1][1] * y2
        print(f"    block {j}: ({x2[2*j]},{x2[2*j+1]}) = ({y1},{y2})  ->  "
              f"{H[0][0]}*{y1}+{H[0][1]}*{y2} = {r1} -> {r1 % M} ({c[2*j]}),   "
              f"{H[1][0]}*{y1}+{H[1][1]}*{y2} = {r2} -> {r2 % M} ({c[2*j+1]})")
    print("    ...")
    _wrap("CIPHERTEXT:", c)
    assert c == encrypt(plaintext, a, b, vig_key, hill_key)
    return c


def decrypt_verbose(ciphertext, a, b, vig_key, hill_key, show=8):
    """Same result as decrypt(), but prints every step."""
    _line("DECRYPTION - STEP 0: check the key and invert the layers in reverse order")
    check_key(a, b, vig_key, hill_key)
    H = hill_matrix(hill_key)
    det = (H[0][0] * H[1][1] - H[0][1] * H[1][0]) % M
    di = modinv(det)
    Hi = mat_inv(H)
    K = clean(vig_key)
    print(f"    det(H) = {det},  det^-1 mod 26 = {di}   (check: {det}*{di} mod 26 = {(det * di) % M})")
    print(f"    H^-1 = det^-1 * adj(H) mod 26 = {Hi[0]} / {Hi[1]}")
    prod = [[sum(H[r][k] * Hi[k][cc] for k in range(2)) % M for cc in range(2)] for r in range(2)]
    print(f"    check H * H^-1 mod 26 = {prod[0]} / {prod[1]}  (identity)")
    print(f"    a^-1 mod 26 = {modinv(a)}   (check: {a}*{modinv(a)} mod 26 = {(a * modinv(a)) % M})")
    ct = clean(ciphertext)
    _wrap(f"ciphertext ({len(ct)} letters):", ct)

    _line("STEP 1 - undo HILL:  (y1, y2) = H^-1 * (c1, c2) mod 26")
    y = hill_apply(ct, Hi)
    for j in range(min(show // 2, len(ct) // 2)):
        c1, c2 = ord(ct[2 * j]) - 65, ord(ct[2 * j + 1]) - 65
        r1 = Hi[0][0] * c1 + Hi[0][1] * c2
        r2 = Hi[1][0] * c1 + Hi[1][1] * c2
        print(f"    block {j}: ({ct[2*j]},{ct[2*j+1]}) = ({c1},{c2})  ->  {r1 % M} ({y[2*j]}), {r2 % M} ({y[2*j+1]})")
    print("    ...")
    _wrap("after Hill^-1:", y)

    _line("STEP 2 - undo VIGENERE:  x = (y - key[i mod len]) mod 26")
    x = vig_dec(y, vig_key)
    for i in range(min(show, len(y))):
        yv, kv = ord(y[i]) - 65, ord(K[i % len(K)]) - 65
        print(f"    pos {i}: {y[i]} - {K[i % len(K)]} = {yv}-{kv} = {(yv - kv) % M} -> {x[i]}")
    print("    ...")
    _wrap("after Vigenere^-1:", x)

    _line("STEP 3 - undo AFFINE:  t = a^-1 * (x - b) mod 26")
    ai = modinv(a)
    t = affine_dec(x, a, b)
    for ch in x[:show]:
        v = ord(ch) - 65
        print(f"    {ch}={v:>2}: {ai}*({v}-{b}) = {(ai * (v - b)) % M:>2} -> {chr(65 + (ai * (v - b)) % M)}")
    print("    ...")
    _wrap("DECRYPTED (cleaned plaintext; trailing X may be padding):", t)
    assert t == decrypt(ciphertext, a, b, vig_key, hill_key)
    return t


if __name__ == "__main__":
    KEY = dict(a=7, b=11, vig_key="NETWORKSECURITY", hill_key="HILL")
    msg = "Meet me at the server room at midnight and bring the firewall logs"
    ct = encrypt_verbose(msg, **KEY)
    pt = decrypt_verbose(ct, **KEY)
    print("\nRESULT: decrypted text equals the cleaned plaintext ->", pt == pad(clean(msg)))
