import json
import numpy as np
import torch
import torch.nn as nn
import librosa
import os

# Prevent duplicate library errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define your mapping (adjust according to your classes)
mapping = [None, 'gt', 'diffwave', 'melgan', 'parallel_wave_gan', 'wavegrad', 'wavenet', 'wavernn']

class AudioClassificationModel(nn.Module):
    def __init__(self):
        super(AudioClassificationModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(130 * 13, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, len(mapping) - 1)  # Output layer size based on your mapping
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

def load_model(model_path, device):
    """Load the trained model."""
    model = AudioClassificationModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, max_len=130):
    """Extract multiple features from an audio file."""
    try:
        signal, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None, None
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T
    
    # Extract Zero-Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    
    # Extract Pitch
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    pitch = [p[np.argmax(m)] for p, m in zip(pitches, magnitudes) if m.any()]
    avg_pitch = np.mean(pitch) if pitch else 0

    # Pad or truncate MFCC to fixed length
    if len(mfcc) < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    
    return mfcc, zcr, avg_pitch

def hybrid_predict(model, audio_path, device, threshold=0.6):
    """Make a prediction using the model and additional metrics."""
    mfcc_features, zcr, pitch = extract_features(audio_path)
    if mfcc_features is None:
        return "Error extracting features"
    
    features_tensor = torch.tensor(mfcc_features, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probabilities) + 1  # Adjust index to match mapping
        confidence = probabilities[predicted_index - 1]
    
    predicted_label = mapping[predicted_index]
    result = "Real Voice" if predicted_label == "gt" else "AI Generated"
    
    print(f"Model Prediction: {predicted_label} ({result}) with confidence: {confidence:.2f}")
    print(f"Zero-Crossing Rate: {zcr:.4f}, Pitch: {pitch:.2f}")

    adjusted_confidence = confidence * 0.7 if pitch < 80 else confidence

    if adjusted_confidence < threshold:
        print("Low confidence. Classified as: Unknown")
        return "Unknown"
    else:
        return result

if __name__ == "__main__":
    MODEL_PATH = r"C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\audio_classification_model1.pth"
    AUDIO_PATH = r"C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\14_Solving_overfitting_in_neural_network\\code\\19_227_000004_000003.wav"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)
    
    if model is not None:
        result = hybrid_predict(model, AUDIO_PATH, device)
        print(f"Final Prediction: {result}")
    else:
        print("Model loading failed.")
