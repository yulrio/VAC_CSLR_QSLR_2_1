# Panduan AI (AGENTS.md)

## Konteks Proyek
- Repositori: `VAC_CSLR_QSLR_2_1`.
- Fokus Saat Ini: Memahami arsitektur Vanilla Continuous Sign Language Recognition (CSLR).
- Direktori Catatan: Seluruh penjelasan arsitektur disimpan di `/home/rajo/Documents/My Researchs/VAC_CSLR_QSLR_2_1/learning/`.

## Aturan Komunikasi & Dokumentasi
1. **Bahasa**: Gunakan Bahasa Indonesia.
2. **Gaya Penjelasan Catatan (Markdown)**:
   - Detail, bertahap (step-by-step).
   - Gunakan contoh perhitungan angka sederhana/abstrak untuk algoritma kompleks (seperti Conv1D, MaxPool, LSTM).
   - Gunakan diagram Mermaid untuk memvisualisasikan perubahan dimensi tensor (Shape Tracker).
3. **Peringatan Dimensi PyTorch**:
   - Ingat aturan dimensi `nn.Conv1d` PyTorch: `[Batch, Channel, Time]`.
   - Selalu lacak proses `reshape`, `transpose`, dan `permute` sebelum dan sesudah lapisan konvolusi/rekursif.
4. **Alur Belajar yang Sedang Berjalan**:
   - Selesai: Conv2D (ResNet/Masked_BN), Conv1D (Temporal Local).
   - Selanjutnya: BiLSTM (Temporal Global), CTC Loss.
