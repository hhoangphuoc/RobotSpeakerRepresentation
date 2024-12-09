import os
import re
import string
import pandas as pd
import torch
import librosa
import subprocess
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-large')

# Function to embed audio and save embeddings to CSV
#def embed_and_save(audio_file, output_path):
    
#    embedding_df.to_csv(output_path, sep=",", index=False)
    
#    print(f"Embeddings saved to {output_path}")

# Directory containing audio files
#directory = "/Users/hideki/wav2vec"
#directory = "/home/khiet/venvironments/env4/kt/audio1"
#directory = "/home/khiet/venvironments/env4/kt/audio-emote"
#directory = "/home/khiet/venvironments/env4/kt/audio-r2d2"

directory = "/home/khiet/venvironments/env4/kt/audio-montreal"
embeddings_txt = "/home/khiet/venvironments/env4/kt/embeddings_montreal.txt"

column_names = []
for i in range(1, 513):
    column_names.append('col' + str(i))
df = pd.DataFrame()
#df.columns = column_names

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".wav"):  # Adjust file extension as necessary
        file_path = os.path.join(directory, filename)
        print(filename)
        # Generate output path for embeddings CSV
        output_file = os.path.splitext(filename)[0] + ".csv"
        output_path = os.path.join(directory, output_file)

        command = ['sox', '--i', '-D', file_path]

        # Execute the command and capture the output
        duration_str = subprocess.check_output(command).decode('utf-8').strip()

        # Convert duration string to float
        duration = float(duration_str)

        if duration > 0.3:
            # Embed audio and save embeddings to CSV
            audio_file = file_path
            #embed_and_save(file_path, output_path)
            audio, _ = librosa.load(audio_file, sr=16000)
            inputs = feature_extractor(audio, padding=True, return_tensors="pt")
            embeddings = model(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    
            embedding_np = embeddings.detach().cpu().numpy()
            #embedding_np = embedding_np.transpose()
            embedding_df = pd.DataFrame(embedding_np)
            #column_names = []

            # Generate column names
            #for i in range(1, 513):
            #    column_names.append('col' + str(i))
            #embedding_df_t = embedding_df.transpose()
            #embedding_df.columns = column_names
            base_name = os.path.basename(audio_file)
            print(base_name)
            # 6_disgust_m.wav
            emo = base_name
            emo = re.sub('.*disgust.*', 'disgust', emo)
            emo = re.sub('.*anger.*', 'anger', emo)
            emo = re.sub('.*fear.*', 'fear', emo)
            emo = re.sub('.*happiness.*', 'happiness', emo)
            emo = re.sub('.*pleasure.*', 'pleasure', emo)
            emo = re.sub('.*pain.*', 'pain', emo)
            emo = re.sub('.*neutral.*', 'neutral', emo)
            emo = re.sub('.*sadness.*', 'sadness', emo)
            emo = re.sub('.*surprise.*', 'surprise', emo)

            gender = base_name
            gender = re.sub(r'.*_m.*', 'm', gender)
            gender = re.sub(r'.*_f.*', 'f', gender)

            #speaker = base_name
            speaker = re.sub(r'^([0-9]+)_.*_.\.wav', r'\1', base_name)
            speaker = "montreal_" + speaker

            embedding_df["filename"] = base_name
            embedding_df["lab_emo"] = emo
            embedding_df["lab_database"] = "montreal"
            embedding_df["lab_gender"] = gender
            embedding_df["lab_speaker"] = speaker

            df = df._append(embedding_df, ignore_index=True)

            print(f"Embeddings saved")



column_names.append("filename")
column_names.append("lab_emo")
column_names.append("lab_database")
column_names.append("lab_gender")
column_names.append("lab_speaker")
df.columns = column_names

df.to_csv(embeddings_txt, sep=",", index=False)



