import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from trilayer_cipher import *
r = json.load(open("results.json"))
A = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
x = np.arange(26)
fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.bar(x - 0.2, np.array(r["freq_plain"]) * 100, 0.4, label="Plaintext (IC = %.3f)" % r["ic"]["plaintext"], color="#2f6f9f")
ax.bar(x + 0.2, np.array(r["freq_cipher"]) * 100, 0.4, label="TriLayer ciphertext (IC = %.3f)" % r["ic"]["ciphertext"], color="#d98b2b")
ax.set_xticks(x); ax.set_xticklabels(list(A), fontsize=8); ax.set_ylabel("Frequency (%)"); ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig("fig_freq.png", dpi=200); plt.close()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
n = [d["n"] for d in r["attackA"]]
a1.plot(n, [d["full"] * 100 for d in r["attackA"]], "o-", color="#2f6f9f", label="Full recovery")
a1.plot(n, [d["acc"] * 100 for d in r["attackA"]], "s--", color="#d98b2b", label="Letters correct")
a1.set_xscale("log"); a1.set_xticks([200,400,1000,2000,4000]); a1.set_xticklabels(["200","400","1000","2000","4000"]); a1.minorticks_off(); a1.set_xlabel("Ciphertext length (letters)"); a1.set_ylabel("%"); a1.set_title("Attack A: frequency analysis", fontsize=9)
a1.legend(fontsize=7, frameon=False)
m = [d["m"] for d in r["attackB"]]
a2.plot(m, [d["full"] * 100 for d in r["attackB"]], "o-", color="#2f6f9f")
a2.set_xlabel("Known plaintext (letters)"); a2.set_title("Attack B: known plaintext", fontsize=9)
for a in (a1, a2): a.spines[["top", "right"]].set_visible(False); a.set_ylim(-3, 103)
plt.tight_layout(); plt.savefig("fig_success.png", dpi=200); plt.close()

KEY = dict(a=7, b=11, vig_key="NETWORKSECURITY", hill_key="HILL")
msg = "Meet me at the server room at midnight and bring the firewall logs"
t0 = pad(clean(msg)); t1 = affine_enc(t0, 7, 11); t2 = vig_enc(t1, KEY["vig_key"]); t3 = hill_apply(t2, hill_matrix("HILL"))
print(t0); print(t1); print(t2); print(t3); assert t3 == encrypt(msg, **KEY)
L = 15
print("nominal bits", math.log2(312) + math.log2(157248) + L * math.log2(26), "effective", math.log2(157248) + L * math.log2(26))
print(math.log2(157248), math.log2(312), L*math.log2(26))
