from collections import Counter, defaultdict
from typing import Dict, List, Optional

import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .evaluators.evaluation import ExplanationEvaluation
from .explainers.explanation import Explanation


def get_colormap(format):

    if format == "blue_red":
        return sns.diverging_palette(240, 10, as_cmap=True)
    elif format == "white_purple":
        return sns.light_palette("purple", as_cmap=True)
    elif format == "purple_white":
        return sns.light_palette("purple", as_cmap=True, reverse=True)
    elif format == "white_purple_white":
        colors = ["white", "purple", "white"]
        return LinearSegmentedColormap.from_list("diverging_white_purple", colors)
    else:
        raise ValueError(f"Unknown format {format}")


def get_dataframe(explanations: List[Explanation]) -> pd.DataFrame:
    """Convert explanations into a pandas DataFrame.

    Args:
        explanations (List[Explanation]): list of explanations

    Returns:
        pd.DataFrame: explanations in table format. The columns are the tokens and the rows are the explanation scores, one for each explainer.
    """
    scores = {e.explainer: e.scores for e in explanations}
    scores["Token"] = explanations[0].tokens
    table = pd.DataFrame(scores).set_index("Token").T
    return table


def deduplicate_column_names(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    column_counts = Counter(df_copy.columns)

    new_columns = list()
    seen_names = defaultdict(int)
    for column in df_copy.columns:

        count = column_counts[column]
        if count > 1:
            new_columns.append(f"{column}_{seen_names[column]}")
            seen_names[column] += 1
        else:
            new_columns.append(column)

    df_copy.columns = new_columns
    return df_copy


def style_heatmap(df: pd.DataFrame, subsets_info: List[Dict]):
    """Style a pandas DataFrame as a heatmap.

    Args:
        df (pd.DataFrame): a pandas DataFrame
        subsets_info (List[Dict]): a list of dictionaries containing the style information for each subset of the DataFrame. Each dictionary should contain the following keys: vmin, vmax, cmap, axis, subset. See https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Building-Styles for more information.

    Returns:
        pd.io.formats.style.Styler: a styled pandas DataFrame
    """

    style = df.style
    for si in subsets_info:
        style = style.background_gradient(**si)
    return style.format("{:.2f}")


def show_table(
    explanations: List[Explanation], remove_first_last: bool, style: str, **style_kwargs
):
    """Format explanation scores into a colored table.

    Args:
        explanations (List[Explanation]): list of explanations
        apply_style (bool): apply color to the table of explanation scores
        remove_first_last (bool): do not visualize the first and last tokens, typically CLS and EOS tokens

    Returns:
        pd.DataFrame: a colored (styled) pandas dataframed
    """

    # Get scores as a pandas DataFrame
    table = get_dataframe(explanations)

    if remove_first_last:
        table = table.iloc[:, 1:-1]

    # add count as prefix for duplicated tokens
    table = deduplicate_column_names(table)
    if not style:
        return table.style.format("{:.2f}")

    if style == "heatmap":
        subset_info = {
            "vmin": style_kwargs.get("vmin", -1),
            "vmax": style_kwargs.get("vmax", 1),
            "cmap": style_kwargs.get("cmap", get_colormap("blue_red")),
            "axis": None,
            "subset": None,
        }
        return style_heatmap(table, [subset_info])
    else:
        raise ValueError(f"Style {style} is not supported.")


def show_evaluation_table(
    explanation_evaluations: List[ExplanationEvaluation],
    style: Optional[str],
) -> pd.DataFrame:
    """Format evaluation scores into a colored table.

    Args:
        explanation_evaluations (List[ExplanationEvaluation]): a list of evaluations of explanations
        apply_style (bool): color the table of evaluation scores

    Returns:
        pd.DataFrame: a colored (styled) pandas dataframe of evaluation scores
    """

    # Flatten to a tabular format: explainers x evaluation metrics
    flat = list()
    for evaluation in explanation_evaluations:
        d = dict()
        d["Explainer"] = evaluation.explanation.explainer
        for metric_output in evaluation.evaluation_outputs:
            d[metric_output.metric.SHORT_NAME] = metric_output.value
        flat.append(d)

    table = pd.DataFrame(flat).set_index("Explainer")

    if not style:
        return table.format("{:.2f}")

    if style == "heatmap":

        subsets_info = list()

        # TODO: we use here the first explainer evaluation, assuming every evaluation in the list will have the same list of outputs
        outputs = explanation_evaluations[0].evaluation_outputs
        for output in outputs:

            vmin = output.metric.MIN_VALUE
            vmax = output.metric.MAX_VALUE
            best_value = output.metric.BEST_VALUE

            if vmin == -1 and vmax == 1 and best_value == 0:
                cmap = get_colormap("white_purple_white")
            else:
                cmap = (
                    get_colormap("purple_white")
                    if output.metric.LOWER_IS_BETTER
                    else get_colormap("white_purple")
                )

            subsets_info.append(
                dict(
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    axis=1,
                    subset=[output.metric.SHORT_NAME],
                )
            )

        style = style_heatmap(table, subsets_info)
        return style
    else:
        raise ValueError(f"Style {style} is not supported.")


# def _style_evaluation(table: pd.DataFrame) -> pd.DataFrame:

#     """Apply style to evaluation scores.

#     Args:
#         table (pd.DataFrame): the evaluation scores as pandas DataFrame

#     Returns:
#         pd.io.formats.style.Styler: a colored and styled pandas dataframe of evaluation scores
#     """

#     table_style = table.style.background_gradient(
#         axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
#     )

#     show_higher_cols, show_lower_cols = list(), list()

#     # Color differently the evaluation measures for which "high score is better" or "low score is better"
#     # Darker colors mean better performance
#     for evaluation_measure in self.evaluators + self.class_based_evaluators:
#         if evaluation_measure.SHORT_NAME in table.columns:
#             if evaluation_measure.BEST_SORTING_ASCENDING == False:
#                 # Higher is better
#                 show_higher_cols.append(evaluation_measure.SHORT_NAME)
#             else:
#                 # Lower is better
#                 show_lower_cols.append(evaluation_measure.SHORT_NAME)

#     if show_higher_cols:
#         table_style.background_gradient(
#             axis=1,
#             cmap=EVALUATION_PALETTE,
#             vmin=-1,
#             vmax=1,
#             subset=show_higher_cols,
#         )

#     if show_lower_cols:
#         table_style.background_gradient(
#             axis=1,
#             cmap=EVALUATION_PALETTE_REVERSED,
#             vmin=-1,
#             vmax=1,
#             subset=show_lower_cols,
#         )
#     return table_style
