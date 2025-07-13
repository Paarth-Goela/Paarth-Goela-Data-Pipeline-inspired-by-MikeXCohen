"""
Neural Data Analysis Pipeline - Backend
Based on Mike X Cohen's analysis style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
import mne
from mne.time_frequency import tfr_morlet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import os
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

class NeuralDataAnalyzer:
    """
    A comprehensive neural data analysis pipeline following Mike X Cohen's approach
    """
    def __init__(self):
        self.raw = None
        self.epochs = None
        self.events = None
        self.event_id = None
        self.tfr = None
        self.pac_results = None
        self.power_features = None
        self.phase_features = None
        self.feature_names = None
        self.ml_model = None
        self.ml_results = None

    def load_sample_data(self):
        """
        Load MNE sample data for demonstration
        """
        try:
            from mne.datasets import sample
            data_path = sample.data_path()
            raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
            self.raw = mne.io.read_raw_fif(raw_fname, preload=True)
            self.raw.pick_types(meg=False, eeg=True, stim=False, exclude='bads')
            print("Sample data loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            return False

    def load_data(self, filepath, file_type=None):
        """
        Load EEG/LFP data from file
        """
        try:
            if file_type is None:
                file_type = os.path.splitext(filepath)[1].lower().lstrip('.')
            if file_type == 'edf':
                self.raw = mne.io.read_raw_edf(filepath, preload=True)
            elif file_type == 'fif':
                self.raw = mne.io.read_raw_fif(filepath, preload=True)
            elif file_type == 'set':
                self.raw = mne.io.read_raw_eeglab(filepath, preload=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            self.raw.pick_types(meg=False, eeg=True, stim=False, exclude='bads')
            print(f"Loaded data from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def preprocess_data(self, l_freq=1.0, h_freq=40.0, resample_freq=None, reference='average', notch_freq=None):
        """
        Preprocess EEG data with filtering, re-referencing, and resampling
        """
        if self.raw is None:
            raise ValueError("No data loaded. Please load data first.")
        self.raw = self.raw.copy()
        if notch_freq is not None:
            self.raw.notch_filter(notch_freq, picks='eeg')
            print(f"Applied notch filter at {notch_freq} Hz")
        if l_freq is not None or h_freq is not None:
            self.raw.filter(l_freq, h_freq, picks='eeg')
            print(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")
        if reference == 'average':
            self.raw.set_eeg_reference('average', projection=True)
            self.raw.apply_proj()
            print("Applied average reference")
        elif reference == 'mastoids':
            print("Mastoid referencing not implemented (requires mastoid channels)")
        if resample_freq is not None:
            self.raw.resample(resample_freq)
            print(f"Resampled data to {resample_freq} Hz")
        return True

    def create_epochs(self, tmin=-0.5, tmax=1.0, baseline=(-0.2, 0.0)):
        """
        Create epochs from events
        """
        if self.raw is None:
            raise ValueError("No data loaded. Please load data first.")
        if self.events is None or len(self.events) < 2:
            print("⚠️  Not enough events found. Injecting fake events for debugging.")
            sfreq = self.raw.info['sfreq']
            n_events_to_inject = 5
            event_times_fake = np.arange(1, n_events_to_inject + 1) * 2 * sfreq
            self.events = np.column_stack([
                event_times_fake.astype(int),
                np.zeros(n_events_to_inject, dtype=int),
                np.tile([1, 2], int(np.ceil(n_events_to_inject / 2)))[:n_events_to_inject]
            ])
            self.event_id = {'condition_1': 1, 'condition_2': 2}
            print(f"[DEBUG] Forced {len(self.events)} fake events for debugging.")
        self.epochs = mne.Epochs(self.raw, self.events, self.event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, reject_by_annotation=True)
        print(f"Created {len(self.epochs)} epochs")
        print(f"[DEBUG] Number of epochs: {len(self.epochs)}")
        print(f"[DEBUG] Shape of self.epochs.get_data(): {self.epochs.get_data().shape}")
        print(f"[DEBUG] Events used (first 10): \n{self.events[:min(10, len(self.events))]}")
        print(f"Event types: {self.event_id}")
        if len(self.epochs) < 2:
            print("[ERROR] Only 1 or fewer epochs created. TFR will fail. This usually means:")
            print(" - Your data annotations didn't create enough events")
            print(" - Filtering or rejection removed most epochs")
            print(" - Even fake events didn't inject properly")
            print("→ Try reloading data or reducing tmin/tmax.")
            raise ValueError("Not enough epochs (need at least 2). Aborting to prevent 3D TFR error.")
        return True

    def time_frequency_analysis(self, freqs=None, n_cycles=7):
        """
        Compute time-frequency representation (TFR) using Morlet wavelets
        """
        if self.epochs is None:
            raise ValueError("No epochs created. Please create epochs first.")
        if freqs is None:
            freqs = np.logspace(np.log10(4), np.log10(40), 20)
        self.tfr = tfr_morlet(self.epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)
        print("Time-frequency analysis completed")
        print(f"[DEBUG] TFR data shape: {self.tfr.data.shape}")
        return True

    def extract_power_features(self, freq_bands=None, time_window=(0.0, 0.5)):
        """
        Extract power features from TFR for ML
        """
        if self.tfr is None:
            raise ValueError("No TFR computed. Please run time-frequency analysis first.")
        if freq_bands is None:
            freq_bands = {
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 40)
            }
        sfreq = self.epochs.info['sfreq']
        times = self.tfr.times
        tmin_idx = np.argmin(np.abs(times - time_window[0]))
        tmax_idx = np.argmin(np.abs(times - time_window[1]))
        features = []
        feature_names = []
        for ch in range(self.tfr.data.shape[1]):
            for band, (fmin, fmax) in freq_bands.items():
                fmin_idx = np.argmin(np.abs(self.tfr.freqs - fmin))
                fmax_idx = np.argmin(np.abs(self.tfr.freqs - fmax))
                band_power = self.tfr.data[:, ch, fmin_idx:fmax_idx+1, tmin_idx:tmax_idx+1].mean(axis=(2, 3))
                features.append(band_power)
                feature_names.append(f"ch{ch}_{band}")
        self.power_features = np.stack(features, axis=-1)
        self.feature_names = feature_names
        print(f"Extracted {self.power_features.shape[-1]} power features")
        return True

    def compute_phase_amplitude_coupling(self, low_freq=(4,8), high_freq=(30,40), method='tort'):
        """
        Compute PAC using the Tort method (default)
        """
        if self.epochs is None:
            raise ValueError("No epochs created. Please create epochs first.")
        data = self.epochs.get_data()
        sfreq = self.epochs.info['sfreq']
        n_epochs, n_channels, n_times = data.shape
        pac_results = {}
        for ch in range(n_channels):
            ch_data = data[:, ch, :].flatten()
            # Bandpass filter for phase
            b, a = butter(2, [low_freq[0]/(sfreq/2), low_freq[1]/(sfreq/2)], btype='band')
            phase_data = filtfilt(b, a, ch_data)
            phase = np.angle(hilbert(phase_data))
            # Bandpass filter for amplitude
            b, a = butter(2, [high_freq[0]/(sfreq/2), high_freq[1]/(sfreq/2)], btype='band')
            amp_data = filtfilt(b, a, ch_data)
            amp = np.abs(hilbert(amp_data))
            # Tort Modulation Index
            n_bins = 18
            phase_bins = np.linspace(-np.pi, np.pi, n_bins+1)
            digitized = np.digitize(phase, phase_bins) - 1
            mean_amp = np.array([amp[digitized == i].mean() for i in range(n_bins)])
            mean_amp /= mean_amp.sum()
            mi = (np.log(n_bins) + np.sum(mean_amp * np.log(mean_amp + 1e-8))) / np.log(n_bins)
            pac_results[f'ch{ch}'] = mi
        self.pac_results = pac_results
        print("PAC analysis completed")
        return True

    def classify_conditions(self, test_size=0.3):
        """
        Simple ML classification using Random Forest
        """
        if self.power_features is None or self.epochs is None:
            raise ValueError("Features or epochs missing for classification.")
        X = self.power_features.reshape(self.power_features.shape[0], -1)
        y = self.epochs.events[:, 2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        self.ml_model = clf
        self.ml_results = {'report': report, 'confusion_matrix': cm}
        print("Classification completed")
        return True

    # Add more methods as needed for ERP, connectivity, source localization, etc. 