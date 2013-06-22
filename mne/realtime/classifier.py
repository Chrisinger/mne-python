# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from sklearn.base import TransformerMixin
from mne.fiff import pick_types
from mne.time_frequency import multitaper_psd


class RtClassifier:

    """
    TODO: complete docstring ...

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, estimator):

        self.estimator = estimator

    def fit(self, X, y):

        self.estimator.fit(X, y)
        return self

    def predict(self, X):

        result = self.estimator.predict(X)

        return result


class Scaler(TransformerMixin):

    def __init__(self, info):
        self.info = info

    def transform(self, epochs_data):

        picks_list = [pick_types(self.info, meg='mag', exclude='bads'),
                      pick_types(self.info, eeg='True', exclude='bads'),
                      pick_types(self.info, meg='grad', exclude='bads')]

        for pick_one in picks_list:
                ch_mean = epochs_data[:, pick_one, :].mean(axis=1)[:, None, :]
                epochs_data[:, pick_one, :] -= ch_mean

        return epochs_data


class PSDEstimator(TransformerMixin):

    def __init__(self, info):
        self.info = info

    def transform(self, data):
        return multitaper_psd(data)
