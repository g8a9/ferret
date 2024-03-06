""""
Code adapted from https://github.com/emanuel-metzenthin/Lime-For-Time
Adaptation of the LIME algorithm for time series data.
Jonas Hering, Emanuel Metzenthin, Alexander Zenner

We adapted the code to work with unequal width splits to account for word-level audio segmentation.

"""

import numpy as np
import sklearn
from lime import explanation
from lime import lime_base
import math
import logging


class TSDomainMapper(explanation.DomainMapper):
    def __init__(self, signal_names, num_slices, is_multivariate):
        """Init function.
        Args:
            signal_names: list of strings, names of signals
        """
        self.num_slices = num_slices
        self.signal_names = signal_names
        self.is_multivariate = is_multivariate

    def map_exp_ids(self, exp, **kwargs):
        # in case of univariate, don't change feature ids
        if not self.is_multivariate:
            return exp

        names = []
        for _id, weight in exp:
            # from feature idx, extract both the pair number of slice
            # and the signal perturbed
            nsignal = int(_id / self.num_slices)
            nslice = _id % self.num_slices
            signalname = self.signal_names[nsignal]
            featurename = "%d - %s" % (nslice, signalname)
            names.append((featurename, weight))
        return names


class LimeTimeSeriesExplainer(object):
    """Explains time series classifiers."""

    def __init__(
        self,
        kernel_width=25,
        verbose=False,
        class_names=None,
        feature_selection="auto",
        signal_names=["not specified"],
    ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
            signal_names: list of strings, names of signals
        """

        # exponential kernel
        def kernel(d):
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.signal_names = signal_names

    def explain_instance(
        self,
        timeseries_instance,
        classifier_fn,
        num_slices,
        labels=(1,),
        top_labels=None,
        num_features=10,
        num_samples=5000,
        model_regressor=None,
        replacement_method="mean",
        splits=None,  # list of dicts with start, end and word. # MODIFIED
    ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).
        As distance function DTW metric is used.

        Args:
            time_series_instance: time series to be explained.
            classifier_fn: classifier prediction probability function,
                which takes a list of d arrays with time series values
                and outputs a (d, k) numpy array with prediction
                probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            num_slices: Defines into how many slices the time series will
                be split up
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
            splits: list of dicts with start, end and word. # MODIFIED
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if splits != None:
            """
            If provided, we use the splits as interpretable representations
            """
            (
                permutations,
                predictions,
                distances,
            ) = self.__data_labels_distances_word_splits(
                timeseries_instance,
                classifier_fn,
                num_samples,
                num_slices,
                splits,
                replacement_method,
            )
        else:
            print(
                "Using Equal width splits as in the original library: https://github.com/emanuel-metzenthin/Lime-For-Time"
            )
            permutations, predictions, distances = self.__data_labels_distances(
                timeseries_instance,
                classifier_fn,
                num_samples,
                num_slices,
                replacement_method,
            )

        is_multivariate = len(timeseries_instance.shape) > 1

        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]

        domain_mapper = TSDomainMapper(self.signal_names, num_slices, is_multivariate)

        ret_exp = explanation.Explanation(
            domain_mapper=domain_mapper, class_names=self.class_names
        )
        ret_exp.predict_proba = predictions[0]

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        for label in labels:
            (
                ret_exp.intercept[int(label)],
                ret_exp.local_exp[int(label)],
                ret_exp.score,
                ret_exp.local_pred,
            ) = self.base.explain_instance_with_data(
                permutations,
                predictions,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
        return ret_exp

    def __data_labels_distances(
        cls,
        timeseries,
        classifier_fn,
        num_samples,
        num_slices,
        replacement_method="mean",
    ):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing slices from the
        time series and replacing them with other data points (specified by
        replacement_method: mean over slice range, mean of entire series or
        random noise). Then predicts with the classifier.

        Args:
            timeseries: Time Series to be explained.
                it can be a flat array (univariate)
                or (num_signals, num_points) (multivariate)
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear
                model (perturbation + original time series)
            num_slices: how many slices the time series will be split into
                for discretization.
            replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of slices in the time series. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: distance between the original instance and
                    each perturbed instance
        """

        def distance_fn(x):
            return (
                sklearn.metrics.pairwise.pairwise_distances(
                    x, x[0].reshape([1, -1]), metric="cosine"
                ).ravel()
                * 100
            )

        num_channels = 1
        len_ts = len(timeseries)
        if len(timeseries.shape) > 1:  # multivariate
            num_channels, len_ts = timeseries.shape

        values_per_slice = math.ceil(len_ts / num_slices)
        deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries.copy()]

        for i, num_inactive in enumerate(deact_per_sample, start=1):
            logging.info("sample %d, inactivating %d", i, num_inactive)
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(
                features_range, num_inactive, replace=False
            )
            num_channels_to_perturb = np.random.randint(1, num_channels + 1)

            channels_to_perturb = np.random.choice(
                range(num_channels), num_channels_to_perturb, replace=False
            )

            logging.info("sample %d, perturbing signals %r", i, channels_to_perturb)

            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0

            tmp_series = timeseries.copy()

            for idx in inactive_idxs:
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len_ts)

                if replacement_method == "mean":
                    # use mean of slice as inactive
                    perturb_mean(tmp_series, start_idx, end_idx, channels_to_perturb)
                elif replacement_method == "noise":
                    # use random noise as inactive
                    perturb_noise(tmp_series, start_idx, end_idx, channels_to_perturb)
                elif replacement_method == "total_mean":
                    # use total series mean as inactive
                    perturb_total_mean(
                        tmp_series, start_idx, end_idx, channels_to_perturb
                    )
                elif replacement_method == "silence":
                    # NEW - MODIFIED - We also include the silence as a possible replacement
                    perturb_zeroing(tmp_series, start_idx, end_idx, channels_to_perturb)
            original_data.append(tmp_series)

        predictions = classifier_fn(np.array(original_data))

        # create a flat representation for features
        perturbation_matrix = perturbation_matrix.reshape(
            (num_samples, num_channels * num_slices)
        )
        distances = distance_fn(perturbation_matrix)

        return perturbation_matrix, predictions, distances

    def __data_labels_distances_word_splits(
        cls,
        timeseries,
        classifier_fn,
        num_samples,
        num_slices,
        splits,
        replacement_method="mean",
    ):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing slices from the
        time series and replacing them with other data points (specified by
        replacement_method: mean over slice range, mean of entire series or
        random noise). Then predicts with the classifier.

        Args:
            timeseries: Time Series to be explained.
                it can be a flat array (univariate)
                or (num_signals, num_points) (multivariate)
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear
                model (perturbation + original time series)
            num_slices: how many slices the time series will be split into
                for discretization.
            replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of slices in the time series. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: distance between the original instance and
                    each perturbed instance
        """

        def distance_fn(x):
            return (
                sklearn.metrics.pairwise.pairwise_distances(
                    x, x[0].reshape([1, -1]), metric="cosine"
                ).ravel()
                * 100
            )

        num_channels = 1
        len_ts = len(timeseries)
        if len(timeseries.shape) > 1:  # multivariate
            num_channels, len_ts = timeseries.shape

        assert len(splits) == num_slices, "splits must be of length num_slices"

        # values_per_slice = math.ceil(len_ts / num_slices)
        deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries.copy()]

        for i, num_inactive in enumerate(deact_per_sample, start=1):
            logging.info("sample %d, inactivating %d", i, num_inactive)
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(
                features_range, num_inactive, replace=False
            )
            num_channels_to_perturb = np.random.randint(1, num_channels + 1)

            channels_to_perturb = np.random.choice(
                range(num_channels), num_channels_to_perturb, replace=False
            )

            logging.info("sample %d, perturbing signals %r", i, channels_to_perturb)

            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0

            tmp_series = timeseries.copy()

            for idx in inactive_idxs:
                # Rather than equally sized slices, we use the word-level splits
                start_idx = splits[idx]["start"]
                end_idx = splits[idx]["end"]
                end_idx = min(end_idx, len_ts)

                if replacement_method == "mean":
                    # use mean of slice as inactive
                    perturb_mean(tmp_series, start_idx, end_idx, channels_to_perturb)
                elif replacement_method == "noise":
                    # use random noise as inactive
                    perturb_noise(tmp_series, start_idx, end_idx, channels_to_perturb)
                elif replacement_method == "total_mean":
                    # use total series mean as inactive
                    perturb_total_mean(
                        tmp_series, start_idx, end_idx, channels_to_perturb
                    )
                elif replacement_method == "silence":
                    # NEW - MODIFIED - We also include the silence as a possible replacement
                    perturb_zeroing(tmp_series, start_idx, end_idx, channels_to_perturb)
            original_data.append(tmp_series)

        predictions = classifier_fn(np.array(original_data))

        # create a flat representation for features
        perturbation_matrix = perturbation_matrix.reshape(
            (num_samples, num_channels * num_slices)
        )
        distances = distance_fn(perturbation_matrix)

        return perturbation_matrix, predictions, distances


def perturb_total_mean(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = m.mean()
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = m[chan].mean()


def perturb_zeroing(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = 0
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = 0


def perturb_mean(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.mean(m[start_idx:end_idx])
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.mean(m[chan][start_idx:end_idx])


def perturb_noise(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.random.uniform(m.min(), m.max(), end_idx - start_idx)
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.random.uniform(
            m[chan].min(), m[chan].max(), end_idx - start_idx
        )
