import json
import numpy as np
import torch
import torch.nn as nn
import librosa
import os

# Prevent duplicate OpenMP initialization errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define your mapping
mapping = ['diffwave', 'gt', 'melgan', 'parallel_wave_gan', 'wavegrad', 'wavenet', 'wavernn']

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
        self.output = nn.Linear(64, len(mapping))
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
    """Load the saved model."""
    model = AudioClassificationModel()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def extract_mfcc_from_audio(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, max_len=130):
    """Extract MFCC features from an audio file."""
    signal, sample_rate = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    
    # Ensure MFCC features have a consistent length
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc

def predict_mfcc_with_probabilities(model, mfcc_features, device, threshold=0.5):
    """Make predictions using the model and return probabilities."""
    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(mfcc_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        max_prob = probabilities[predicted_index]
        predicted_label = mapping[predicted_index]

        # Check if the highest probability is below the threshold
        if max_prob < threshold:
            return "Unknown", "Unknown", probabilities
        
        # Classify as Real or AI Generated Voice
        if predicted_label == 'gt':
            category = "Real Voice"
        else:
            category = "AI Generated Voice"

    return predicted_label, category, probabilities

if __name__ == "__main__":
    # Paths for the model and audio file
    MODEL_PATH = r"C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\audio_classification_model1.pth"
    AUDIO_PATH = r"C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\14_Solving_overfitting_in_neural_network\\code\\B.mp3"
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model
    model = load_model(MODEL_PATH, device)
    
    # Extract MFCC features from the new audio file
    new_mfcc = extract_mfcc_from_audio(AUDIO_PATH)
    
    # Make predictions and get probabilities
    predicted_label, category, probabilities = predict_mfcc_with_probabilities(model, new_mfcc, device, threshold=0.6)
    
    # Print results
    print(f"Predicted Class: {predicted_label}")
    print(f"Category: {category}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{mapping[i]}: {prob:.4f}")
