# ChemNLP-Score: Prototipe Awal Sistem Penilaian Esai Kimia Berbasis NLP (Web Streamlit)

# 1. Instalasi awal (jalankan di lingkungan Jupyter/Colab/terminal untuk setup)
# !pip install streamlit transformers sentence-transformers nltk

import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Cek dan unduh resource NLTK hanya jika belum tersedia
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

setup_nltk()

# 2. Persiapan Model NLP
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Model semantic similarity

# 3. Jawaban Ideal dan Rubrik
jawaban_ideal = "Ikatan ionik terjadi karena transfer elektron dari logam ke nonlogam. Ikatan kovalen terjadi karena atom saling berbagi pasangan elektron. Contohnya adalah NaCl (ionik) dan H2O (kovalen)."
rubrik = {
    "konsep_dasar": 2,
    "istilah_ilmiah": 1,
    "penalaran": 2,
    "contoh": 1,
    "koherensi": 1
}

# 4. Fungsi untuk membersihkan teks
def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('indonesian') and word not in string.punctuation]
    return " ".join(tokens)

# 5. Fungsi Penilaian Semantic
def skor_semantic(jawaban_siswa, jawaban_ideal):
    emb1 = model.encode(jawaban_siswa, convert_to_tensor=True)
    emb2 = model.encode(jawaban_ideal, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(emb1, emb2).item()
    return sim_score  # nilai antara 0 - 1

# 6. Fungsi Penilaian Manual Berdasar Kata Kunci
def cek_kata_kunci(jawaban, kata_kunci):
    return any(kw in jawaban for kw in kata_kunci)

# 7. Fungsi Penilaian Lengkap
def nilai_jawaban(jawaban):
    skor = 0
    umpan_balik = []
    bersih = clean_text(jawaban)

    if cek_kata_kunci(bersih, ["transfer", "berbagi", "elektron", "logam", "nonlogam"]):
        skor += 2
        umpan_balik.append("✅ Konsep dasar ikatan dijelaskan dengan baik.")
    else:
        umpan_balik.append("⚠ Konsep dasar kurang lengkap.")

    if cek_kata_kunci(bersih, ["ionik", "kovalen"]):
        skor += 1
        umpan_balik.append("✅ Istilah ilmiah digunakan dengan tepat.")
    else:
        umpan_balik.append("⚠ Istilah ilmiah belum digunakan secara eksplisit.")

    if cek_kata_kunci(bersih, ["berbeda", "perbedaan", "dibanding"]):
        skor += 2
        umpan_balik.append("✅ Ada upaya membandingkan jenis ikatan.")
    else:
        umpan_balik.append("⚠ Penalaran perbandingan tidak jelas.")

    if cek_kata_kunci(bersih, ["nacl", "h2o"]):
        skor += 1
        umpan_balik.append("✅ Contoh relevan diberikan.")
    else:
        umpan_balik.append("⚠ Contoh belum disebutkan secara spesifik.")

    if len(jawaban.split('.')) > 1:
        skor += 1
        umpan_balik.append("✅ Koherensi argumen baik.")
    else:
        umpan_balik.append("⚠ Penjelasan kurang terstruktur.")

    return skor, umpan_balik

# 8. Antarmuka Web Streamlit
st.set_page_config(page_title="ChemNLP Score", layout="centered")
st.title("🧪 ChemNLP-Score")
st.subheader("Penilaian Esai Otomatis untuk Topik Ikatan Kimia")

jawaban_input = st.text_area("Masukkan jawaban esai siswa di bawah ini:", height=200)

if st.button("🔍 Nilai Jawaban"):
    if jawaban_input.strip() == "":
        st.warning("Silakan masukkan jawaban terlebih dahulu.")
    else:
        skor, feedback = nilai_jawaban(jawaban_input)
        sim = skor_semantic(jawaban_input, jawaban_ideal)

        st.markdown(f"### Skor Total: {skor}/7")
        st.progress(skor / 7)

        st.markdown("### Umpan Balik:")
        for fb in feedback:
            st.write(fb)

        st.markdown(f"### Skor Kemiripan Semantik: {sim:.2f} (0: tidak mirip, 1: sangat mirip)")
