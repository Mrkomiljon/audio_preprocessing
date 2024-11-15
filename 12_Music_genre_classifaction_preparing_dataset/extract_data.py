import json
import os
import math
import librosa

# Constants
DATASET_PATH = r"C:\\Users\\GOOD\\Desktop\\TEST-2024\\Synthetic-Voice-Detection-Vocoder-Artifacts-main\\LibriSeVoc"
JSON_PATH = "data_LibriSeVoc1.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from dataset and saves them to a JSON file."""
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    current_label = 2  # Start labeling other folders from 2

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            semantic_label = os.path.basename(dirpath).strip()  # Remove extra spaces
            print(f"DEBUG: Processing folder '{semantic_label}'")  # Debug statement

            # Assign labels
            if semantic_label.lower() == "gt":
                label = 1  # Real Voice
                print(f"Assigned label 1 to '{semantic_label}'")
            else:
                label = current_label  # Assign incrementing labels for other classes
                current_label += 1  # Increment label for the next folder
                print(f"Assigned label {label} to '{semantic_label}'")

            data["mapping"].append(semantic_label)

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(label)
                            print(f"Processed segment {d + 1} of file {f} with label {label}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save data to JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print(f"Data saved to {json_path}")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
