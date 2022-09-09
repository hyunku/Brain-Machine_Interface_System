import joblib as joblib
from pygame.version import ver
import torch
from torch import nn
import mne
from brainflow.board_shim import BoardIds, BoardShim
from config import load_parameters
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# parameters
device = 'cpu' # for pytorch inference
board_id = BoardIds.CYTON_DAISY_BOARD.value
sfreq = BoardShim.get_sampling_rate(board_id)
eeg_channels = [1, 2, 3, 4, 5, 6]
n_channels = 6
filt = {'l_freq': .5, 'h_freq': 45, 'method': 'iir'}
pretrained_model = "res/lin_model.pkl"  # pretrained model load
load_parameters(globals())

# parameters, not loaded by config.json
ch_names = ['F7', 'F3', 'Fp1', 'Fp2', 'F4', 'F8']


def filtering_bandpass_update(data):
    # Function about extract filtered data from data measured by 0.5sec and extract bandpass data from data measuerd by 3sec

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')  # make info

    raw = mne.io.RawArray(data, info, verbose=False)    # make mne object

    # data is filtered by 0.5sec ( raw + bandpass) -> need to rewrite code on bandpass data to measure 3sec
    raw_df = raw.filter(l_freq=60, h_freq=.5, picks='eeg', method='iir').to_data_frame().drop(columns=['time'])  # 0.5 ~ 60
    delta_band_df = raw.filter(l_freq=4, h_freq=.5, picks='eeg', method='iir').to_data_frame().drop(columns=['time'])  # 0.5 ~ 4
    theta_band_df = raw.filter(l_freq=8, h_freq=4, picks='eeg', method='iir').to_data_frame().drop(columns=['time'])  # 4 ~ 8
    alpha_band_df = raw.filter(l_freq=12, h_freq=8, picks='eeg', method='iir').to_data_frame().drop(columns=['time'])  # 8 ~12
    beta_band_df = raw.filter(l_freq=30, h_freq=12, picks='eeg', method='iir').to_data_frame().drop(columns=['time'])  # 12 ~ 30
    gamma_band_df = raw.filter(l_freq=45, h_freq=30, picks='eeg', method='iir').to_data_frame().drop(columns=['time'])  # 30 ~ 45

    raw_filtered = pd.concat([raw_df, delta_band_df, theta_band_df, alpha_band_df, beta_band_df, gamma_band_df], axis=1)

    # col names
    ch_names_arr = np.char.array(['F7', 'F3', 'Fp1', 'Fp2', 'F4', 'F8'])
    band_labels_arr = np.char.array(["-origin", "-Delta", "-Theta", "-Alpha", "-Beta", "-Gamma"])
    ch_band_labels = ch_names_arr[np.newaxis, :] + band_labels_arr[:, np.newaxis]
    new_cols = ch_band_labels.flatten()

    raw_filtered.columns = new_cols
    print(raw_filtered)

    return raw_filtered   # used in eeg plot


def Inference(data):
    model = joblib.load(pretrained_model)

    # load trained weights here
    print(f"load '{pretrained_model}' here")

    sig = model.predict(data)

    return sig
