"""Shared numeric functions from experiments/h1/extract.py."""
import numpy as np
import librosa

def mfcc_features(x,config):
    mel=librosa.feature.melspectrogram(y=x,sr=16000,n_fft=config['n_fft'],hop_length=config['hop_length'],
        win_length=config['n_fft'],window=config['window'],center=config['center'],pad_mode=config['pad_mode'],
        power=config['power'],n_mels=config['n_mels'],fmin=config['fmin'],fmax=config['fmax'],
        norm=config['mel_norm'],htk=config['htk'])
    db=librosa.power_to_db(mel,ref=1.0,top_db=config['top_db'])
    mf=librosa.feature.mfcc(S=db,n_mfcc=config['n_mfcc'],dct_type=config['dct_type'],norm=config['dct_norm'])
    return np.r_[mf.mean(axis=1),mf.std(axis=1,ddof=0)].astype(np.float32)
