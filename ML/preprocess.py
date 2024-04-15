import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb
import wfdb.processing

ANNOTATION_SYMBOLS = ['N', 'A', 'V'] # Other beat annotations are not of interest
NON_BEAT_SYMBOLS = ['[', ']', '!', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^',
                    '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
NORMALISED_WIDTH = 2047
DEFAULT_PATH = 'mitdb'

"""Class to store 3 QRS complexes and their respective beat annotation symbols."""
class Triplet:
    def __init__(self, signal, annotations):
        self.signal = signal
        self.annotations = annotations

    def __str__(self):
        return f"Signal: {self.signal}, Annotations: {self.annotations}"

    def __repr__(self):
        return f"Signal: {self.signal}, Annotations: {self.annotations}"
    
class Preprocessor:
    def __init__(self, path):
        self.path = path
        self.triplets = []
        self.accepted = {}
        self.rejected = {}
    
    """Get a list of all record names in the path."""
    def get_record_names(self):
        record_names = [f.split('.')[0] for f in os.listdir(self.path) if f.endswith('.hea')]
        return record_names

    """Load a single ECG record's MLII signal and (filtered) annotations."""
    def load_ecg_record(self, record_name, test=False):
        record_path = self.path + '/' + record_name
        record = wfdb.rdrecord(record_path)
        
        ann_locs, ann_symbols = None, None
        
        # Load the annotations, if any
        # If not, use WFDB to identify QRS complexes and use them as annotations
        if os.path.isfile(record_path + '.atr') and not test:
            annotation = wfdb.rdann(record_path, 'atr')
            ann_locs = annotation.sample
            ann_symbols = annotation.symbol

            # Filter the annotations to only the beat annotations
            ann_locs = np.array([loc for loc, symbol in zip(ann_locs, ann_symbols) if symbol not in NON_BEAT_SYMBOLS])
            ann_symbols = np.array([symbol for symbol in ann_symbols if symbol not in NON_BEAT_SYMBOLS])
        else:
            # TODO: Implement R peak detection
            pass
        
        # Filter the signal to only MLII (this signal type is assumed to exist)
        signal = record.p_signal[:, np.argwhere(np.array(record.sig_name) == 'MLII').flatten()]

        # Get the sampling frequency
        fs = record.fs

        return signal, fs, ann_locs, ann_symbols

    """Plot an ECG record."""
    def plot_ecg_record(self, signal, fs, ann_locs, ann_symbols):
        # Create the time vector
        time = np.arange(signal.shape[0]) / fs

        # Plot the first 5 seconds of the ECG with the annotation symbols
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal)
        if ann_locs is not None and ann_symbols is not None:
            for ann_loc, ann_symbol in zip(ann_locs, ann_symbols):
                ann_time = ann_loc / fs
                if ann_time > 5:
                    break
                plt.text(ann_loc / fs, signal[ann_loc], ann_symbol, color='red', weight='bold')
                plt.axvline(x=ann_loc / fs, color='orange', linestyle='--', linewidth=0.5)
        plt.xlim([0, 5])
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.title('ECG Record')
        plt.grid(True)
        plt.show()

    """Normalise a signal to be of length NORMALISED_WIDTH, height between 0 and 1 and centered around 0."""
    def normalise(self, signal):

        plt.clf()
        plt.plot(signal)
        plt.show()

        # Normalise the signal amplitude to be between 0 and 1
        try:
            signal = wfdb.processing.normalize_bound(signal, lb=0, ub=1)
        except:
            print("Error normalising signal:")
            print(signal)
        # Shift the signal vertically so that the start and end points are centered around 0
        signal = signal - ((signal[0] + signal[-1]) / 2)

        # Extend the signal if it is shorter than NORMALISED_WIDTH:
        #  - At the start/end, add points sampled from a Gaussian distribution with mean equal
        #    to the first/last point of the signal and stdev = 0.002
        #  - The signal is extended symmetrically so that the QRS complex is in the middle
        #  - The signal is guaranteed to not be longer than NORMALISED_WIDTH, so no shortening needed
        if len(signal) < NORMALISED_WIDTH:
            start_extension = np.random.normal(signal[0], 0.0025, (NORMALISED_WIDTH - len(signal)) // 2)
            end_extension = np.random.normal(signal[-1], 0.0025, NORMALISED_WIDTH - len(signal) - len(start_extension))
            signal = np.concatenate((start_extension, signal.flatten(), end_extension))
        
        return signal

    """Check if a triplet is valid."""
    def is_valid_triplet(self, signal, annotations):
        # A triplet is valid if:
        #  - The middle QRS complex is one of 'N', 'A', or 'V'
        #  - The length of the triplet is less than or equal to NORMALISED_WIDTH (2048 sample points)
        return annotations[1] in ANNOTATION_SYMBOLS and len(signal) <= NORMALISED_WIDTH and signal.size > 0

    """Preprocess a single ECG record.

    Returns:
    - List of valid triplets
    - Dictionary of accepted triplet annotation symbols
    - Dictionary of rejected triplet annotation symbols
    """
    def preprocess_ecg_record(self, signal, ann_locs, ann_symbols):
        triplets = []

        accepted = {}
        rejected = {}

        # Try to add the first triplet (special case: start point = signal start)
        triplet_signal = signal[:(ann_locs[2] + ann_locs[3]) // 2]
        triplet_symbols = ann_symbols[:3]
        first_key = ''.join(triplet_symbols)

        if self.is_valid_triplet(triplet_signal, triplet_symbols):
            triplets.append(Triplet(self.normalise(triplet_signal), triplet_symbols))
            accepted[first_key] = accepted.get(first_key, 0) + 1
        else:
            rejected[first_key] = rejected.get(first_key, 0) + 1

        # Iterate through the rest of the signal to find all valid triplets
        for i in range(0, len(ann_locs) - 5):
            start = (ann_locs[i] + ann_locs[i + 1]) // 2
            end = (ann_locs[i + 3] + ann_locs[i + 4]) // 2

            triplet_signal = signal[start:end]
            triplet_symbols = ann_symbols[i + 1:i + 4]
            key = ''.join(triplet_symbols)

            if self.is_valid_triplet(triplet_signal, triplet_symbols):
                triplets.append(Triplet(self.normalise(triplet_signal), triplet_symbols))
                accepted[key] = accepted.get(key, 0) + 1
            else:
                rejected[key] = rejected.get(key, 0) + 1

        return triplets, accepted, rejected

    """Preprocess all ECG records in the path."""
    def preprocess_ecg_records(self):
        all_triplets = []
        all_accepted = {}
        all_rejected = {}

        record_names = self.get_record_names()
        for record_name in sorted(record_names):
            signal, fs, ann_locs, ann_symbols = self.load_ecg_record(record_name)
            triplets, accepted, rejected = self.preprocess_ecg_record(signal, ann_locs, ann_symbols)
            all_triplets.extend(triplets)

            for key, value in accepted.items():
                all_accepted[key] = all_accepted.get(key, 0) + value
            for key, value in rejected.items():
                all_rejected[key] = all_rejected.get(key, 0) + value

        # Store the triplets, accepted and rejected triplets
        self.triplets = all_triplets
        self.accepted = all_accepted
        self.rejected = all_rejected

        return all_triplets, all_accepted, all_rejected

    """Return the number of triplets grouped by the middle annotation symbol."""
    def count_by_middle_annotation(self, triplets):
        counts = {}
        for triplet in triplets:
            counts[triplet.annotations[1]] = counts.get(triplet.annotations[1], 0) + 1
        return counts

def main():
    preproc = Preprocessor(DEFAULT_PATH)
    triplets, accepted, rejected = preproc.preprocess_ecg_records()
    print(f"Number of triplets: {len(triplets)}")

    # Count the number of accepted triplets grouped by the second character in the key string
    accepted_counts = {}
    for key, value in accepted.items():
        accepted_counts[key[1]] = accepted_counts.get(key[1], 0) + value
    print(f"Accepted triplets: {accepted_counts} (total {sum(accepted_counts.values())})")
    rejected_counts = {}
    for key, value in rejected.items():
        rejected_counts[key[1]] = rejected_counts.get(key[1], 0) + value
    print(f"Rejected triplets: {rejected_counts} (total {sum(rejected_counts.values())})")

    # Plot the first ECG record
    # signal, fs, ann_locs, ann_symbols = load_ecg_record('100', DEFAULT_PATH)
    # plot_ecg_record(signal, fs, ann_locs, ann_symbols)

# main()