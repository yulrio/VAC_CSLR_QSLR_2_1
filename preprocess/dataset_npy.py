import numpy as np

# Ganti path sesuai lokasi file .npy Anda
npy_path = "QSLR2024/dev_info.npy"

# Membaca file .npy (dictionary)
data = np.load(npy_path, allow_pickle=True).item()

# # Contoh: print semua key dan value
# for key, value in data.items():
#   print(f"{key}: {value}")

target_fileid = "112_alikhlas_1_2_3"
for item in data.values():
    if isinstance(item, dict) and item.get('fileid') == target_fileid:
        print(item)
        break