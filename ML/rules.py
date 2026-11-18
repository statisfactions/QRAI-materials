"""Data and tree drawing for Worksheet 10.2, Rules by Machine (Fall 2026).

Students read the sklearn code in the worksheet but not this file. It provides:

    dengueso_df, gorrux_df, tirannica_df   the three diseases (same data as Spring 2026)
    show_tree(model, X, y)                 draw a fitted DecisionTreeClassifier the way
                                           WS 10.1's tool drew trees: a question in each box,
                                           training counts in each leaf, "Right: x of n",
                                           and the rule in words
    DecisionTreeClassifier, plot_tree, pd, np, plt

The data generators are Spring's hidden setup cell, unchanged (seed 42, same order), so the
numbers in the solution notebook carry over. Dengueso's rule (weight below 202.5) matches
WS 10.1's dengueso in ML/xeno.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeClassifier, plot_tree

SICK = "#E8833A"
WELL = "#4C8BC9"


def _frame(e, hc, a, w, ht, sick):
    return pd.DataFrame({
        "eyes": e, "hair": hc, "antennae": a, "weight": w, "height": ht,
        "eyes_orange": (e == "orange").astype(int),
        "hair_pink": (hc == "pink").astype(int),
        "antennae_7": (a == 7).astype(int),
        "xhealth": np.where(sick, "sick", "well"),
    })


def _make():
    r = np.random.default_rng(42)
    out = []
    for n in [100, 120, 150]:
        w = r.uniform(190, 215, n).round(1)
        e = r.choice(["purple", "orange"], n)
        hc = r.choice(["pink", "blue"], n)
        a = r.choice([7, 8], n)
        ht = r.uniform(100, 135, n).round(1)
        if n == 100:
            sick = w < 202.5
        elif n == 120:
            sick = (e == "orange") | ((e == "purple") & (ht > 120))
        else:
            base = (a == 7) | (e == "orange")
            flip = r.random(n) < 0.10
            sick = np.where(base, ~flip, flip)
        out.append(_frame(e, hc, a, w, ht, sick))
    return out


dengueso_df, gorrux_df, tirannica_df = _make()

# 0/1 columns read back as the question 10.1 asked. sklearn sends "<= 0.5" (the 0s) left,
# so for these the "yes" branch is the right child.
_YESNO = {"eyes_orange": "eyes are orange", "hair_pink": "hair is pink",
          "antennae_7": "antennae = 7"}


def _same_leaves(t, i):
    """A split whose two leaves predict the same thing changes nothing: draw it as one leaf."""
    l, r = t.children_left[i], t.children_right[i]
    if t.children_left[l] != -1 or t.children_left[r] != -1:
        return False
    return np.argmax(t.value[l][0]) == np.argmax(t.value[r][0])


def _node(tree, names, i):
    t = tree
    if t.children_left[i] == -1 or _same_leaves(t, i):
        return None
    name = names[t.feature[i]]
    left, right = t.children_left[i], t.children_right[i]
    if name in _YESNO:
        return {"label": _YESNO[name] + "?", "yes": right, "no": left}
    cut = t.threshold[i]
    return {"label": name + " below " + ("%.1f" % cut) + "?", "yes": left, "no": right}


def _leaf_text(model, i):
    counts = model.tree_.value[i][0]
    classes = list(model.classes_)
    # sklearn may store fractions; convert to whole counts with n_node_samples
    n = model.tree_.n_node_samples[i]
    frac = counts / counts.sum()
    n_sick = int(round(frac[classes.index("sick")] * n)) if "sick" in classes else 0
    n_well = n - n_sick
    pred = classes[int(np.argmax(counts))]
    return pred, n_sick, n_well


def _box(x, y, text, face, width, height, bold=False):
    plt.gca().add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                       boxstyle="round,pad=0.01", facecolor=face,
                                       edgecolor="black", linewidth=1.2))
    plt.text(x, y, text, ha="center", va="center", fontsize=11,
             fontweight="bold" if bold else "normal")


def _arrow(x1, y1, x2, y2, label):
    plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    plt.text((x1 + x2) / 2 + (0.03 if x2 > x1 else -0.03), (y1 + y2) / 2, label,
             ha="left" if x2 > x1 else "right", va="center", fontsize=10, style="italic")


def _draw(model, names, i, x, y, spread):
    node = _node(model.tree_, names, i)
    if node is None:
        pred, s, w = _leaf_text(model, i)
        _box(x, y, "predict " + pred + "\n" + str(s) + " sick, " + str(w) + " well",
             SICK if pred == "sick" else WELL, 0.2, 0.15, bold=True)
        return
    _box(x, y, node["label"], "#F2F2F2", 0.3 if spread > 0.2 else 0.24, 0.13)
    for branch, side in [("yes", -1), ("no", 1)]:
        cx, cy = x + side * spread, y - 0.3
        _arrow(x + side * 0.05, y - 0.07, cx, cy + 0.08, branch)
        _draw(model, names, node[branch], cx, cy, 0.13)


def _words(model, names, i, inner=False):
    node = _node(model.tree_, names, i)
    if node is None:
        return "predict " + _leaf_text(model, i)[0]
    text = ("if " + node["label"].rstrip("?") + ", " + _words(model, names, node["yes"], True)
            + "; otherwise " + _words(model, names, node["no"], True))
    return "(" + text + ")" if inner else text


def show_tree(model, X, y):
    """Draw a fitted tree in WS 10.1's style, with its score on X and y."""
    names = list(X.columns)
    depth = model.get_depth()
    if depth > 2:
        print("This drawing shows trees of depth 1 or 2. Use plot_tree for deeper trees.")
        return
    right = int((model.predict(X) == np.asarray(y)).sum())
    plt.figure(figsize=(9, 2.6 + 1.6 * max(depth, 1)))
    _draw(model, names, 0, 0.5, 0.9, 0.25)
    plt.xlim(0, 1)
    plt.ylim(0.88 - 0.3 * max(depth, 1) - 0.12, 1.0)
    plt.axis("off")
    plt.title("The machine's tree on the " + str(len(y)) + " training cases. Right: "
              + str(right) + " of " + str(len(y)), fontsize=12)
    plt.show()
    print("The rule in words: " + _words(model, names, 0))
    t = model.tree_
    if any(t.children_left[i] != -1 and _same_leaves(t, i) for i in range(t.node_count)):
        print("(The machine asked one more question on one side, but both answers predicted the"
              " same thing, so the drawing leaves it out.)")
