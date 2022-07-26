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

import datasets
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import pytreebank


set_names = ["train", "validation", "test"]


def main(output_dir, normalize=False):
    """
    Author: Eliana Pastor
    Adapted from https://github.com/BoulderDS/evaluating-human-rationales/blob/66402dbe8ccdf8b841c185cd8050b8bdc04ef3d2/scripts/download_and_process_sst.py#L20
    """
    print(f"Output dir: {output_dir} normalization: {normalize}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    data_ptb = datasets.load_dataset("sst", "ptb")

    for set_name in set_names:
        rationales = []
        for i in range(0, len(data_ptb[set_name])):
            item = pytreebank.create_tree_from_string(data_ptb[set_name][i]["ptb_tree"])
            rationales.append(list(get_sst_rationale(item, normalize=normalize)))

        rationales_df = pd.DataFrame([rationales]).T
        rationales_df.columns = ["rationale"]
        rationales_df.to_csv(os.path.join(output_dir, f"sst_rationales_{set_name}.csv"))
        with open(
            os.path.join(output_dir, f"sst_rationales_{set_name}.pickle"), "wb"
        ) as handle:
            pickle.dump(rationales, handle)


def get_leaves(tree):
    leaves = []
    if len(tree.children) > 0:
        for child in tree.children:
            leaves += get_leaves(child)
    else:
        leaves.append(tree)
    return leaves


def get_sst_rationale(item, normalize=False):
    """
    Author: Eliana Pastor
    Adapted from https://github.com/BoulderDS/evaluating-human-rationales/blob/66402dbe8ccdf8b841c185cd8050b8bdc04ef3d2/scripts/download_and_process_sst.py#L74
    """
    rationale = []
    count_leaves_and_extreme_descendants(item)
    phrases = []
    assemble_rationale_phrases(item, phrases, normalize=normalize)
    for phrase in phrases:
        phrase_rationale = [
            np.abs(normalize_label(phrase.label)) if normalize else np.abs(phrase.label)
        ] * phrase.num_leaves
        rationale.extend(phrase_rationale)
        pass
    if normalize:
        rationale = np.array(rationale)
    else:
        rationale = np.array(rationale) - 2
    return rationale


def explanatory_phrase(tree, normalize=False):
    if len(tree.children) == 0:
        return True
    else:
        if normalize:
            label = normalize_label(tree.label)
            max_descendant = normalize_label(tree.max_descendant)
            min_descendant = normalize_label(tree.min_descendant)
        else:
            label = tree.label
            max_descendant = tree.max_descendant
            min_descendant = tree.min_descendant

        if abs(label) > abs(max_descendant) and abs(label) > abs(min_descendant):
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


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default=os.path.join(os.getcwd(), "data", "sst_data"),
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="specify normalize to normalize labels as in 'Evaluating and Characterizing Human Rationales - Samuel Carton, Anirudh Rathore, Chenhao Tan'",
    )
    args = parser.parse_args()
    main(output_dir=args.name_output_dir, normalize=args.normalize)
