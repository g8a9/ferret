import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pydub
import torch
from datasets import Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """

    return (
        np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio.channels)
        )
        / (1 << (8 * audio.sample_width - 1)),
        audio.frame_rate,
    )


def print_log(*args):
    # This is just a wrapper to easily spot the print :) - I use it to debug
    print(args)


def plot_word_importance_summary(
    df_labels,
    top_k=15,
    at_least=5,
    figsize=(7, 5),
    height=0.5,
    title="",
    ALL="-",
    label_id_to_text=None,
    legend_outside=True,
):
    """
    Plot the importance score for each word, separately for each class value

    Args:
        df_labels: dataframe with the word importance scores
        top_k: number of words to plot
        at_least: minimum number of times a word must appear to be considered
        figsize: size of the figure
        height: height of the bars
        title: title
        ALL: name of the column with the average importance score
        label_id_to_text: id to text label mapping (list)
        legend_outside: if True, the legend is outside the plot
    """

    import matplotlib.pyplot as plt

    import matplotlib.colors as cm

    sel = (
        df_labels.loc[df_labels["count"] > at_least]
        .sort_values(by=[ALL, "count"], ascending=False)
        .head(top_k)
    )

    n_colors = len([i for i in list(sel.T.index)])

    if n_colors < 20:
        cmap = plt.get_cmap("tab20")
        colors = {
            color_label: cm.to_hex(cmap(e))
            for e, color_label in enumerate(list(sel.T.index))
        }
    else:
        cmap = plt.get_cmap("gist_rainbow")

        colors = {
            color_label: cm.to_hex(cmap(1.0 * e / n_colors))
            for e, color_label in enumerate(list(sel.T.index))
        }

    fig, ax = plt.subplots(figsize=figsize)
    stacked = np.zeros(len(sel.T.columns))
    stacked_pos = np.zeros(len(sel.T.columns))
    stacked_neg = np.zeros(len(sel.T.columns))

    # This is just to have the words in the defined order: by highest average importance and count
    ax.barh(
        list(sel.T.columns)[::-1],
        np.zeros(len(sel.T.columns)),
        height=height,
        label=None,
    )

    for class_value_id in sel.T.index:
        if class_value_id not in ["count", ALL]:
            vals = sel.T.loc[class_value_id].values
            labels = sel.T.columns

            if n_colors > 20:
                label_name = (
                    f"{class_value_id}: {label_id_to_text[class_value_id]}"
                    if label_id_to_text is not None
                    else class_value_id
                )

            else:
                label_name = (
                    label_id_to_text[class_value_id]
                    if label_id_to_text is not None
                    else class_value_id
                )

            for i, v in enumerate(vals):
                if v > 0:
                    p = ax.barh(
                        labels[i],
                        v,
                        height=height,
                        left=stacked_pos[i],
                        label=label_name,
                        color=colors[class_value_id],
                    )
                    if n_colors > 20:
                        ax.bar_label(
                            p, labels=[class_value_id], label_type="center", fontsize=6
                        )
                elif v < 0:
                    p = ax.barh(
                        labels[i],
                        v,
                        height=height,
                        left=stacked_neg[i],
                        label=label_name,
                        color=colors[class_value_id],
                    )
                    if n_colors > 20:
                        ax.bar_label(
                            p, labels=[class_value_id], label_type="center", fontsize=6
                        )

            vals = np.nan_to_num(vals)
            stacked += vals
            stacked_neg += np.where(vals > 0, 0, vals)
            stacked_pos += np.where(vals < 0, 0, vals)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # To remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if legend_outside:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=8,
        )
    else:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    if title != "":
        ax.set_title(title, fontsize=9)
    plt.show()
    return fig


def load_dataset_and_model(dataset_name, data_dir, model_dir=None, model_name=None):
    if dataset_name == "FSC":
        ## Load audio

        df = pd.read_csv(f"{data_dir}/data/test_data.csv")
        df["path"] = df["path"].apply(lambda x: os.path.join(data_dir, x))

        dataset = Dataset.from_pandas(df)

        if model_name is None:
            ## Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "superb/wav2vec2-base-superb-ic"
            )
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "superb/wav2vec2-base-superb-ic"
            )
        else:
            ## Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    elif dataset_name == "IEMOCAP":
        # Note that the directory should contain the IEMOCAP folder
        # If the dir is already in the format /IEMOCAP, remove the last part (as for torchaudio loading)
        if os.path.basename(data_dir) == "IEMOCAP":
            data_dir = data_dir.replace(f"{os.path.sep}IEMOCAP", "")

        import torchaudio

        dataset_ta = torchaudio.datasets.IEMOCAP(data_dir, sessions=[1])
        data_dir = data_dir + f"{os.path.sep}IEMOCAP"

        df = pd.DataFrame(
            [dataset_ta.get_metadata(i) for i in range(len(dataset_ta))],
            columns=["path", "sample_rate", "filename", "label", "speaker"],
        )
        df = df[
            [
                "path",
                "label",
            ]
        ]
        df["path"] = df["path"].apply(lambda x: os.path.join(data_dir, x))
        dataset = Dataset.from_pandas(df)

        ## Load model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )

    elif dataset_name == "ITALIC":
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

        if model_dir is None:
            model_dir = f"{str(Path.home())}/ITALIC"

        ## Load model and feature extractor
        model_checkpoint = (
            model_dir
            + "/"
            + "wav2vec2-xls-r-300m-ic-finetuning-hard-speaker/checkpoint-3605"
        )

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, local_files_only=True
        )
        from datasets import load_dataset

        dataset_da = load_dataset(
            "RiTA-nlp/ITALIC", "hard_speaker", use_auth_token=True
        )

        dataset = pd.DataFrame(
            {
                "path": [audio_i["path"] for audio_i in dataset_da["test"]["audio"]],
                "label": dataset_da["test"]["intent"],
            }
        )
        dataset = Dataset.from_pandas(dataset)
    else:
        raise ValueError("Dataset not supported")
    if torch.cuda.is_available():
        model = model.cuda()
    return dataset, model, feature_extractor
