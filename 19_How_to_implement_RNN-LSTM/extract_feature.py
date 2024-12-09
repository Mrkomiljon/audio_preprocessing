import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    """
    Extracts audio features from a given audio file.

    :param file_path: Path to the audio file
    :return: Dictionary containing audio features
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        rmse = np.mean(librosa.feature.rms(y=y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_13 = np.mean(mfccs[12])  # 13th MFCC coefficient

        # Create a feature dictionary
        features = {
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "rmse": rmse,
            "mfcc_13": mfcc_13
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_audio_files(directory):
    """
    Processes all audio files in a directory and extracts features.

    :param directory: Path to the directory containing audio files
    :return: DataFrame containing features for all audio files
    """
    feature_list = []
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # You can adjust this to other formats like .mp3
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                if features:
                    features["file_path"] = file_path
                    feature_list.append(features)
                    file_paths.append(file_path)

    return pd.DataFrame(feature_list)


# Example usage
if __name__ == "__main__":
    # Path to the directory containing audio files
    audio_directory = "C:\\Users\\GOOD\\Desktop\\audio_files"

    # Process audio files and save the features to a CSV
    feature_df = process_audio_files(audio_directory)
    feature_df.to_csv("audio_features.csv", index=False)
    print("Features extracted and saved to 'audio_features.csv'")
