"""

From  https://github.com/BoulderDS/evaluating-human-rationales/blob/66402dbe8ccdf8b841c185cd8050b8bdc04ef3d2/scripts/download_and_process_sst.py
Evaluating and Characterizing Human Rationales
Samuel Carton, Anirudh Rathore, Chenhao Tan

MIT License by Samuel Carton

Copyright (c) 2020 Data Science @ University of Colorado Boulder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import pytreebank


def get_leaves(tree):
    leaves = []
    if len(tree.children) > 0:
        for child in tree.children:
            leaves += get_leaves(child)
    else:
        leaves.append(tree)
    return leaves


def get_sst_rationale(item):
    """
    Author: Eliana Pastor
    Adapted from https://github.com/BoulderDS/evaluating-human-rationales/blob/66402dbe8ccdf8b841c185cd8050b8bdc04ef3d2/scripts/download_and_process_sst.py#L74
    """
    rationale = []
    count_leaves_and_extreme_descendants(item)
    phrases = []
    assemble_rationale_phrases(item, phrases)
    for phrase in phrases:
        phrase_rationale = [np.abs(normalize_label(phrase.label))] * phrase.num_leaves
        rationale.extend(phrase_rationale)
        pass
    rationale = np.array(rationale)
    return rationale


def explanatory_phrase(tree):
    if len(tree.children) == 0:
        return True
    else:
        normalized_label = normalize_label(tree.label)
        normalized_max_descendant = normalize_label(tree.max_descendant)
        normalized_min_descendant = normalize_label(tree.min_descendant)

        if abs(normalized_label) > abs(normalized_max_descendant) and abs(
            normalized_label
        ) > abs(normalized_min_descendant):
            return True
        else:
            return False


def assemble_rationale_phrases(tree, phrases, **kwargs):
    if explanatory_phrase(tree, **kwargs):
        phrases.append(tree)
    else:
        for child in tree.children:
            assemble_rationale_phrases(child, phrases, **kwargs)


def count_leaves_and_extreme_descendants(tree):

    if len(tree.children) == 0:  # if is leaf
        tcount = 1
        tmax = tmin = tree.label
    else:
        tcount = 0
        child_labels = [child.label for child in tree.children]
        tmax = max(child_labels)
        tmin = min(child_labels)

        for child in tree.children:
            ccount, cmax, cmin = count_leaves_and_extreme_descendants(child)
            tcount += ccount
            tmax = max(tmax, cmax)
            tmin = min(tmin, cmin)

    tree.num_leaves = tcount
    tree.max_descendant = tmax
    tree.min_descendant = tmin

    if tree.label == 4:
        _ = None
    return tcount, tmax, tmin


def normalize_label(label):
    return (label - 2) / 2
