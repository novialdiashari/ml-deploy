import csv
from pathlib import Path

from sklearn.datasets import load_diabetes


# 1. Load dataset diabetes dari Scikit-Learn
dataset = load_diabetes()

# Nama-nama fitur
feature_names = dataset.feature_names

# Data input (X) dan target (y)
X = dataset.data
y = dataset.target


# 2. Tentukan lokasi file CSV
# File akan disimpan di folder "data"
data_folder = Path("data")
data_file = data_folder / "diabetes_regression.csv"


# 3. Buat folder "data" jika belum ada
data_folder.mkdir(exist_ok=True)


# 4. Membuka file CSV untuk ditulis
with open(data_file, "w", newline="", encoding="utf-8") as file:

    # Membuat CSV writer
    writer = csv.writer(file)

    # 5. Menulis nama kolom
    header = list(feature_names) + ["target"]
    writer.writerow(header)

    # 6. Menulis data
    for features, target in zip(X, y):
        row = list(features) + [target]
        writer.writerow(row)


# 7. Menampilkan informasi hasil
print("Dataset berhasil disimpan!")
print(f"Lokasi file : {data_file}")
print(f"Jumlah data : {len(y)}")
print(f"Jumlah fitur: {len(feature_names)}")
print(f"Nama fitur  : {', '.join(feature_names)}")