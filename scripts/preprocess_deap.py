#!/usr/bin/env python3
"""
preprocess_deap.py

Convert DEAP *.mat EEG files into .npy format for fast loading.
Also generates a labels.csv containing: sample_id, valence, arousal, subject_id.

DEAP structure:
    Each subject has a .mat file containing:
        data: [40 trials, 40 channels, 8064 samples]
        labels: [40 trials, 4 dims] (valence, arousal, dominance, liking)

Output example:
    data/processed/deap/eeg/subject_01_trial_00.npy
    data/processed/deap/labels.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm

def ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def main(args):
    raw_path = Path(args.raw_folder)
    out_eeg = Path(args.out_folder) / "eeg"
    ensure(out_eeg)

    label_rows = []

    mat_files = sorted(raw_path.glob("*.mat"))
    print("Found DEAP subject files:", len(mat_files))

    sample_counter = 0

    for subject_index, mat_path in enumerate(tqdm(mat_files, desc="Processing DEAP")):
        mat = loadmat(mat_path)
        data = mat["data"]     # shape [40,40,8064]
        labels = mat["labels"] # shape [40,4]

        n_trials = data.shape[0]
        subject_id = subject_index  # simple mapping

        for trial in range(n_trials):
            eeg = data[trial]  # [40,8064]
            val, aro = labels[trial][0], labels[trial][1]

            # Save EEG as .npy
            out_path = out_eeg / f"subject{subject_id:02d}_trial{trial:02d}.npy"
            np.save(out_path, eeg.astype(np.float32))

            label_rows.append([
                sample_counter,
                float(val),
                float(aro),
                subject_id
            ])
            sample_counter += 1

    out_csv = Path(args.out_folder) / "labels.csv"
    pd.DataFrame(label_rows, columns=["sample_id", "valence", "arousal", "subject_id"]).to_csv(out_csv, index=False)

    print("DEAP preprocessing complete.")
    print(f"EEG saved in: {out_eeg}")
    print(f"Label CSV: {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_folder", type=str, required=True)
    p.add_argument("--out_folder", type=str, required=True)
    args = p.parse_args()
    main(args)

