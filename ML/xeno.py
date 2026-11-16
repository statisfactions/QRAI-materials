"""Xeno tree builder for Worksheet 10.1 (Fall 2026).

Students never read or write this code. The worksheet's form cells call five functions:

    get_cases(disease, how_many)      deal training aliens, shown as a colored table
    graph(disease, feature)           sick and well, split by one feature
    build_tree(disease, ...)          draw the students' tree, with training counts in each leaf
    test_tree(disease)                try the tree on 10 aliens it has never seen
    competition(ready)                sumonoma only: the same 100 new aliens for every screen

The diseases follow ML/malady-rules.md (the Xeno game's own rules), with the same columns as
WS 10.2: eyes, hair, antennae, weight, height, xhealth. Training cases come from a fixed pool per
disease, so every screen sees the same first 10 (the lesson plan relies on this). Tests draw fresh
aliens each time, like Xeno; the competition set is fixed so the board compares like with like.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, HTML

SICK = "#E8833A"
WELL = "#4C8BC9"
SICK_LIGHT = "#FBE0CC"
WELL_LIGHT = "#D6E6F5"

DISEASES = ["ague", "botulosis", "dengueso", "sumonoma"]
FEATURES = ["eyes", "hair", "antennae", "weight", "height"]

QUESTIONS = {
    "hair is pink": ("hair", "pink"),
    "hair is blue": ("hair", "blue"),
    "eyes are purple": ("eyes", "purple"),
    "eyes are orange": ("eyes", "orange"),
    "antennae = 7": ("antennae", 7),
    "antennae = 8": ("antennae", 8),
    "weight is below the number": ("weight", None),
    "height is below the number": ("height", None),
}

_SEEDS = {"ague": 101, "botulosis": 202, "dengueso": 303, "sumonoma": 405}
_COMPETITION_SEED = 2026
_pool = {}
_dealt = {}
_trees = {}
_test_rng = np.random.default_rng()
_competition_first = None


def _aliens(disease, n, rng):
    eyes = rng.choice(["purple", "orange"], n)
    hair = rng.choice(["pink", "blue"], n)
    antennae = rng.choice([7, 8], n)
    height = rng.uniform(100, 135, n).round(1)
    if disease == "sumonoma":
        weight = rng.uniform(195, 265, n).round(1)
    else:
        weight = rng.uniform(190, 215, n).round(1)

    if disease == "ague":
        sick = hair == "pink"
    elif disease == "botulosis":
        sick = (eyes == "purple") & (hair == "pink")
    elif disease == "dengueso":
        sick = weight < 202.5
    elif disease == "sumonoma":
        sick = rng.random(n) < np.clip((weight - 212) / 40, 0, 1)
    else:
        raise ValueError("Unknown disease: " + str(disease))

    table = pd.DataFrame({
        "eyes": eyes, "hair": hair, "antennae": antennae,
        "weight": weight, "height": height,
        "xhealth": np.where(sick, "sick", "well"),
    })
    table.index = np.arange(1, n + 1)
    table.index.name = "alien"
    return table


def _check_disease(disease):
    if disease not in DISEASES:
        raise ValueError("The disease must be one of: " + ", ".join(DISEASES))


def _cases(disease):
    if disease not in _dealt:
        get_cases(disease, 10, show=False)
    return _dealt[disease]


def _color_health(value):
    if value == "sick":
        return "background-color: " + SICK_LIGHT
    if value == "well":
        return "background-color: " + WELL_LIGHT
    return ""


def _color_right(value):
    if value == "yes":
        return "color: #1a7f37; font-weight: bold"
    if value == "NO":
        return "color: #c62828; font-weight: bold"
    return ""


def _show(table, health_columns, right_column=None):
    styler = table.style.format({"weight": "{:.1f}", "height": "{:.1f}"})
    styler = styler.map(_color_health, subset=health_columns)
    if right_column is not None:
        styler = styler.map(_color_right, subset=[right_column])
    display(styler)


def get_cases(disease, how_many=10, show=True):
    """Deal the first `how_many` training aliens for a disease and show them."""
    _check_disease(disease)
    how_many = int(how_many)
    if disease not in _pool:
        _pool[disease] = _aliens(disease, 200, np.random.default_rng(_SEEDS[disease]))
    _dealt[disease] = _pool[disease].iloc[:how_many]
    if show:
        cases = _dealt[disease]
        n_sick = int((cases["xhealth"] == "sick").sum())
        display(HTML("<b>" + disease + ": " + str(how_many) + " training cases</b> ("
                     + str(n_sick) + " sick, " + str(how_many - n_sick) + " well)"))
        _show(cases, ["xhealth"])


def graph(disease, feature):
    """Show how sick and well aliens differ on one feature."""
    _check_disease(disease)
    if feature not in FEATURES:
        raise ValueError("The feature must be one of: " + ", ".join(FEATURES))
    cases = _cases(disease)
    sick = cases[cases["xhealth"] == "sick"]
    well = cases[cases["xhealth"] == "well"]

    if feature in ["weight", "height"]:
        plt.figure(figsize=(8, 2.6))
        rng = np.random.default_rng(0)
        plt.scatter(sick[feature], 1 + rng.uniform(-0.12, 0.12, len(sick)), s=90,
                    color=SICK, edgecolor="black", label="sick")
        plt.scatter(well[feature], 0 + rng.uniform(-0.12, 0.12, len(well)), s=90,
                    color=WELL, edgecolor="black", label="well")
        plt.yticks([0, 1], ["well", "sick"])
        plt.ylim(-0.6, 1.6)
        plt.xlabel(feature)
        plt.grid(axis="x", alpha=0.3)
    else:
        values = sorted(cases[feature].unique().tolist(), key=str)
        if len(values) < 2:
            values = sorted(_pool[disease][feature].unique().tolist(), key=str)
        n_sick = [int((sick[feature] == v).sum()) for v in values]
        n_well = [int((well[feature] == v).sum()) for v in values]
        x = np.arange(len(values))
        plt.figure(figsize=(6, 3.6))
        plt.bar(x - 0.2, n_sick, width=0.4, color=SICK, edgecolor="black", label="sick")
        plt.bar(x + 0.2, n_well, width=0.4, color=WELL, edgecolor="black", label="well")
        plt.xticks(x, [feature + ": " + str(v) for v in values])
        plt.ylabel("number of aliens")
        top = max(n_sick + n_well + [1])
        plt.yticks(range(0, top + 1, max(1, (top + 1) // 6)))
    plt.title(disease + ": " + str(len(cases)) + " training cases, by " + feature)
    plt.legend(loc="upper right", bbox_to_anchor=(1.22, 1))
    plt.tight_layout()
    plt.show()


def _question(name, number):
    feature, value = QUESTIONS[name]
    if value is None:
        number = float(number)
        label = feature + " below " + ("%g" % number) + "?"
        return {"feature": feature, "value": None, "number": number, "label": label}
    return {"feature": feature, "value": value, "number": None, "label": name + "?"}


def _answer_yes(node, table):
    if node["value"] is None:
        return (table[node["feature"]] < node["number"]).to_numpy()
    return (table[node["feature"]] == node["value"]).to_numpy()


def _predict(tree, table):
    if isinstance(tree, str):
        return np.full(len(table), tree, dtype=object)
    yes = _answer_yes(tree, table)
    out = np.empty(len(table), dtype=object)
    if yes.any():
        out[yes] = _predict(tree["yes"], table[yes])
    if (~yes).any():
        out[~yes] = _predict(tree["no"], table[~yes])
    return out


def _leaf(choice):
    return "sick" if choice == "predict sick" else "well"


def build_tree(disease, question_1, number_1, if_yes, if_no,
               question_2="(not used)", number_2=0, if_yes_2="predict sick",
               if_no_2="predict well"):
    """Read the form, draw the tree, and remember it for testing."""
    _check_disease(disease)
    ask_yes = if_yes == "ask question 2"
    ask_no = if_no == "ask question 2"
    if ask_yes and ask_no:
        print("Only one branch can ask question 2. Change if_yes or if_no.")
        return
    if (ask_yes or ask_no) and question_2 == "(not used)":
        print("A branch says 'ask question 2', but question_2 is '(not used)'. Choose a question 2.")
        return

    root = _question(question_1, number_1)
    second = None
    if ask_yes or ask_no:
        second = _question(question_2, number_2)
        second["yes"] = _leaf(if_yes_2)
        second["no"] = _leaf(if_no_2)
    root["yes"] = second if ask_yes else _leaf(if_yes)
    root["no"] = second if ask_no else _leaf(if_no)
    _trees[disease] = root

    cases = _cases(disease)
    _draw(disease, root, cases)
    for node in [root, second]:
        if node is not None and node["value"] is None:
            low, high = _pool[disease][node["feature"]].min(), _pool[disease][node["feature"]].max()
            if node["number"] <= low or node["number"] >= high:
                print("Check your number for " + node["feature"] + ": the aliens' "
                      + node["feature"] + "s run from about " + ("%.0f" % low) + " to "
                      + ("%.0f" % high) + ".")
    print("Your rule in words: " + _words(root))


def _words(tree, inner=False):
    if isinstance(tree, str):
        return "predict " + tree
    text = ("if " + tree["label"].rstrip("?") + ", " + _words(tree["yes"], True)
            + "; otherwise " + _words(tree["no"], True))
    return "(" + text + ")" if inner else text


def _box(x, y, text, face, width=0.3, height=0.13, bold=False):
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


def _draw_node(tree, cases, x, y, spread):
    if isinstance(tree, str):
        n_sick = int((cases["xhealth"] == "sick").sum())
        n_well = int((cases["xhealth"] == "well").sum())
        face = SICK if tree == "sick" else WELL
        _box(x, y, "predict " + tree + "\n" + str(n_sick) + " sick, " + str(n_well) + " well",
             face, width=0.2, height=0.15, bold=True)
        return
    _box(x, y, tree["label"], "#F2F2F2", width=0.3)
    yes = _answer_yes(tree, cases)
    for branch, side, part in [("yes", -1, cases[yes]), ("no", 1, cases[~yes])]:
        cx, cy = x + side * spread, y - 0.3
        _arrow(x + side * 0.05, y - 0.07, cx, cy + 0.08, branch)
        _draw_node(tree[branch], part, cx, cy, 0.14)


def _draw(disease, tree, cases):
    right = int((_predict(tree, cases) == cases["xhealth"].to_numpy()).sum())
    depth = 1 + int(not isinstance(tree["yes"], str) or not isinstance(tree["no"], str))
    plt.figure(figsize=(8, 2.6 + 1.6 * depth))
    _draw_node(tree, cases, 0.5, 0.9, 0.25)
    plt.xlim(0, 1)
    plt.ylim(0.88 - 0.3 * depth - 0.12, 1.0)
    plt.axis("off")
    plt.title(disease + ": your tree on the " + str(len(cases)) + " training cases. Right: "
              + str(right) + " of " + str(len(cases)), fontsize=12)
    plt.show()


def _scored(tree, table):
    predicted = _predict(tree, table)
    result = table.copy()
    result = result.rename(columns={"xhealth": "really"})
    result.insert(len(FEATURES), "tree says", predicted)
    result["right?"] = np.where(predicted == table["xhealth"].to_numpy(), "yes", "NO")
    return result, int((result["right?"] == "yes").sum())


def test_tree(disease):
    """Try the tree on 10 aliens it has never seen. Each run draws new aliens."""
    _check_disease(disease)
    if disease not in _trees:
        print("Build your " + disease + " tree first.")
        return
    new = _aliens(disease, 10, _test_rng)
    result, right = _scored(_trees[disease], new)
    display(HTML("<b>" + disease + ": 10 new aliens.</b> Your rule: " + _words(_trees[disease])))
    _show(result, ["tree says", "really"], "right?")
    display(HTML("<p style='font-size:120%'><b>Right: " + str(right) + " of 10 = "
                 + str(right * 10) + "%</b></p>"))


def competition(ready=False):
    """Sumonoma only: the same 100 new aliens for every screen. The first tree tested counts."""
    global _competition_first
    if not ready:
        print("Not run yet. When your screen agrees on its sumonoma tree, tick 'ready' and run this cell.")
        return
    if "sumonoma" not in _trees:
        print("Build your sumonoma tree first.")
        return
    tree = _trees["sumonoma"]
    new = _aliens("sumonoma", 100, np.random.default_rng(_COMPETITION_SEED))
    result, right = _scored(tree, new)
    rule = _words(tree)
    if _competition_first is None:
        _competition_first = (rule, right)
    first_rule, first_right = _competition_first
    if rule != first_rule:
        print("You changed your tree after the competition. Only your first tree counts:")
    display(HTML("<p style='font-size:130%'><b>Competition: " + str(first_right)
                 + " of 100 new aliens right = " + str(first_right) + "%</b></p>"
                 + "<p>Your competition rule: " + first_rule + "</p>"))
    n_sick = int((new["xhealth"] == "sick").sum())
    print("(Of the 100 new aliens, " + str(n_sick) + " were really sick and "
          + str(100 - n_sick) + " were well.)")
