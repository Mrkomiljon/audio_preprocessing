import json
import os
import math
import librosa

# Constants
DATASET_PATH = "genres_original"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from a music dataset and saves them into a JSON file along with genre labels."""

    # Dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Ensure we're processing a genre sub-folder
        if dirpath != dataset_path:
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            # Process all audio files in genre sub-directory
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(data["mapping"].index(semantic_label))
                            print(f"{file_path}, segment: {d + 1}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Yangi JSON faylga dastlabki 10 ta yozuvni saqlash
    first_10_data = {
        "mapping": data["mapping"],
        "labels": data["labels"][:10],
        "mfcc": data["mfcc"][:10]
    }

    with open(json_path, "w") as fp:
        json.dump(first_10_data, fp, indent=4)

    print(f"\nData saved to {json_path}")


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
