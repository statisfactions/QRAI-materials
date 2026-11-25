"""Helpers for Worksheet 11.2, Beyond Accuracy (Fall 2026).

Students read the code in the worksheet but not this file. It adds one thing to rules.py
(WS 10.2): show_tree(model, X, y) now draws trees for any two labels, so it also draws the
Titanic's survived/died trees ("1st or 2nd class?" in place of "ticket_class below 2.5").
The worksheet loads titanic.csv itself and counts with pd.crosstab, as in WS 10.2.

With features female, age, ticket_class on ML/titanic.csv (714 passengers): everyone-died
gets 424 of 714; depth 1 (female) 557, TP 197 FP 64 FN 93 TN 360; depth 2 573, TP 166 FP 17
FN 124 TN 407 (boys under 6.5 survive; women in 1st/2nd class survive, 3rd-class women
predicted died).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rules
from rules import *          # noqa: F401,F403  (tirannica_df, DecisionTreeClassifier, ...)

POSITIVE = ["sick", "survived"]
rules._YESNO["female"] = "female"


def _pos_neg(labels):
    labels = [str(v) for v in labels]
    pos = next(p for p in POSITIVE if p in labels)
    neg = next((v for v in labels if v != pos), {"sick": "well", "survived": "died"}[pos])
    return pos, neg


def _node(tree, names, i):
    node = rules._node(tree, names, i)
    if node is not None:
        node["label"] = (node["label"].replace("ticket_class below 2.5", "1st or 2nd class")
                         .replace("ticket_class below 1.5", "1st class")
                         .replace("ticket_class below 3.5", "any class"))
    return node


def _leaf_box(x, y, text, face):
    plt.gca().add_patch(rules.FancyBboxPatch((x - 0.11, y - 0.1), 0.22, 0.2,
                                             boxstyle="round,pad=0.01", facecolor=face,
                                             edgecolor="black", linewidth=1.2))
    plt.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold")


def _leaf_text(model, i):
    counts = model.tree_.value[i][0]
    classes = [str(c) for c in model.classes_]
    pos, neg = _pos_neg(classes)
    n = model.tree_.n_node_samples[i]
    frac = counts / counts.sum()
    n_pos = int(round(frac[classes.index(pos)] * n))
    return classes[int(np.argmax(counts))], n_pos, n - n_pos, pos, neg


def _draw(model, names, i, x, y, spread):
    node = _node(model.tree_, names, i)
    if node is None:
        pred, a, b, pos, neg = _leaf_text(model, i)
        _leaf_box(x, y, "predict " + pred + "\n" + str(a) + " " + pos + "\n" + str(b) + " " + neg,
                  rules.SICK if pred == pos else rules.WELL)
        return
    rules._box(x, y, node["label"], "#F2F2F2", 0.3 if spread > 0.2 else 0.24, 0.13)
    for branch, side in [("yes", -1), ("no", 1)]:
        cx, cy = x + side * spread, y - 0.3
        rules._arrow(x + side * 0.05, y - 0.07, cx, cy + 0.08, branch)
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
    plt.ylim(0.88 - 0.3 * max(depth, 1) - 0.16, 1.0)
    plt.axis("off")
    plt.title("The machine's tree on the " + str(len(y)) + " training cases. Right: "
              + str(right) + " of " + str(len(y)), fontsize=12)
    plt.show()
    print("The rule in words: " + _words(model, names, 0))
    t = model.tree_
    if any(t.children_left[i] != -1 and rules._same_leaves(t, i) for i in range(t.node_count)):
        print("(The machine asked one more question on one side, but both answers predicted the"
              " same thing, so the drawing leaves it out.)")

