import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

#Drop MP3's inside to get the Genre:

def extract_feature(file_path):
    # Extracting a simple feature - the zero crossing rate
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    feature = np.mean(zero_crossing_rate)
    return feature


data = pd.read_csv('Music.csv')


features = []
for _, row in data.iterrows():
    # Adjust the directory path according to your file locations
    file_path = os.path.join('path/to/your/music/files', row['file_name'])  # Specify the correct path
    feature = extract_feature(file_path)
    genre = row['genre']
    features.append([feature, genre])


features_df = pd.DataFrame(features, columns=['feature', 'genre'])


X = np.array(features_df['feature'].tolist()).reshape(-1, 1)
y = np.array(features_df['genre'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

9
print(classification_report(y_test, predictions))
