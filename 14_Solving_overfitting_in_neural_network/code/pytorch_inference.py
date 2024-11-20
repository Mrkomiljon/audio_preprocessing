import json
import numpy as np
import torch
import torch.nn as nn
import librosa
import os

# Prevent duplicate library errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define your mapping (classes)
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
        self.dropout = nn.Dropout(0.5)
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
    """Load the trained model with error handling."""
    try:
        model = AudioClassificationModel()
        # `weights_only=True` bayrog'ini qo'shdik
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, max_len=130):
    """Extract multiple features from an audio file."""
    try:
        signal, sr = librosa.load(audio_path, sr=22050)
        if len(signal) == 0:
            raise ValueError("Empty audio signal")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T
        
        # Extract Zero-Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
        
        # Extract Pitch using YIN
        pitch = librosa.yin(signal, fmin=50, fmax=300)
        avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0

        # Pad or truncate MFCC to fixed length
        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len, :]

        return mfcc, zcr, avg_pitch
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None

def hybrid_predict(model, audio_path, device, threshold=0.55):
    """Make a prediction using the model and additional metrics."""
    mfcc_features, zcr, pitch = extract_features(audio_path)
    if mfcc_features is None:
        return "Error extracting features"
    
    # Convert MFCC features to a tensor and add batch dimension
    features_tensor = torch.tensor(mfcc_features, dtype=torch.float32, device=device).unsqueeze(0)
    
    try:
        with torch.no_grad():
            # Modeldan prognoz olish
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Har bir sinf uchun ehtimolliklarni ko'rsatish
            print("\nClass Probabilities:")
            for idx, prob in enumerate(probabilities):
                print(f"Class '{mapping[idx]}': {prob:.4f}")

            # Eng yuqori ehtimollikni aniqlash
            predicted_index = np.argmax(probabilities)
            confidence = probabilities[predicted_index]
        
        # Prognoz qilingan label va ularning ishonch darajasi
        predicted_label = mapping[predicted_index]
        
        # Tasniflash natijasi: "Real Voice" yoki "AI Generated"
        if predicted_label == "gt" and confidence > 0.5:
            result = "Real Voice"
        else:
            result = f"AI Generated ({predicted_label})"

        print(f"\nModel Prediction: {predicted_label} ({result}) with confidence: {confidence:.4f}")
        print(f"Zero-Crossing Rate: {zcr:.4f}, Pitch: {pitch:.4f}")

        # Pitch asosida ishonch darajasini moslashtirish
        adjusted_confidence = confidence * 0.7 if pitch < 80 else confidence

        # Agar ishonch darajasi past bo'lsa, "Unknown" deb tasniflash
        if adjusted_confidence < threshold:
            print("Low confidence. Classified as: Unknown")
            return "Unknown"
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error"

if __name__ == "__main__":
    # Model va audio fayl yo'lini kiriting
    MODEL_PATH = r"C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\audio_classification_model.pth"
    AUDIO_PATH = r"C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\14_Solving_overfitting_in_neural_network\\code\\A.mp3"
    
    # CUDA yoki CPU qurilmasini tanlang
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modelni yuklash
    model = load_model(MODEL_PATH, device)
    
    # Model yuklangandan keyin prognoz qilish
    if model is not None:
        result = hybrid_predict(model, AUDIO_PATH, device)
        print(f"\nFinal Prediction: {result}")
    else:
        print("Model loading failed.")
