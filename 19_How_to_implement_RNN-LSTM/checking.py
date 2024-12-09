import pandas as pd

# Datasetni yuklash
file_path = 'C:\\Users\\GOOD\\Desktop\\Komil\\audio_features_dataset.csv'
df = pd.read_csv(file_path)

# Saqlab qolish kerak bo'lgan ustunlar ro'yxati
columns_to_keep = ['file_path', 'zcr', 'spectral_centroid', 'spectral_bandwidth',
                   'spectral_rolloff', 'rmse', 'mfcc_13', 'label']

# Yangi datasetni yaratish
new_df = df[columns_to_keep]

# Yangi datasetni ko'rish
print("Yangi dataset:")
print(new_df.head())

# Yangi datasetni CSV fayliga saqlash
new_file_path = 'C:\\Users\\GOOD\\Desktop\\Komil\\filtered_dataset.csv'
new_df.to_csv(new_file_path, index=False)
print(f"Yangi dataset '{new_file_path}' nomli faylga saqlandi.")
