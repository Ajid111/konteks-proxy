"""
KONTEKS - Word2Vec Server
Pakai model fastText Wikipedia Indonesia dari Facebook Research
Ranking akurat seperti Contexto asli
"""

from flask import Flask, request, jsonify
import os
import gzip
import struct
import numpy as np
from collections import defaultdict
import urllib.request
import threading
import json

app = Flask(__name__)

ROBLOX_SECRET = os.environ.get("ROBLOX_SECRET", "konteks-rahasia-2024")

# ============================================================
# MODEL - load sekali saat server start
# ============================================================
word_vectors = {}   # { kata: np.array }
vocab_list   = []   # semua kata yang ada di model
model_ready  = False
model_error  = None

# Kata-kata yang layak jadi kata rahasia
KATA_LAYAK = [
    "air","api","angin","awan","batu","buku","bulan","bunga","burung","cahaya",
    "daun","desa","gula","gunung","hati","hewan","hutan","hujan","ikan","jalan",
    "kapal","kayu","keluarga","kota","kopi","kucing","laut","langit","lapar",
    "malam","matahari","mimpi","musik","nasi","panas","pantai","pasir","pelangi",
    "pohon","pulau","rumah","sungai","sawah","sekolah","sepi","suara","tangan",
    "tanah","teman","tidur","uang","waktu","warna","api","bola","darah","gelap",
    "garam","emas","harimau","istana","jantung","kabut","kunci","langit","lautan",
    "madu","meja","mimpi","naga","perang","raja","salju","singa","tari","udara",
    "bintang","badai","buah","cinta","dingin","embun","fajar","galaksi","hijau",
    "impian","jubah","kilat","lidah","mawar","nafas","obor","petir","rasa","senja",
    "terang","ular","vokal","wajah","xenon","yoga","zamrud","abadi","bahagia",
    "cahaya","diam","elang","firma","gempa","harap","indah","jujur","karma","lelah",
    "maaf","naluri","obat","pikir","riang","sabar","tegas","ulet","vital","waras",
]

used_words = set()

def cosine_similarity(v1, v2):
    """Hitung cosine similarity antara 2 vektor"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    return dot / norm

def load_model_from_text(filepath):
    """Load model fastText format teks"""
    global word_vectors, vocab_list, model_ready, model_error
    try:
        print(f"Loading model dari {filepath}...")
        count = 0
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            dims = int(header.strip().split()[1]) if header.strip().split() else 300
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                # Hanya simpan kata Indonesia yang bersih
                if not word.isalpha() or len(word) < 2 or len(word) > 20:
                    continue
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vec) == dims:
                        word_vectors[word] = vec
                        vocab_list.append(word)
                        count += 1
                except:
                    continue
        print(f"Model loaded: {count} kata")
        model_ready = True
    except Exception as e:
        model_error = str(e)
        print(f"Error loading model: {e}")

def download_and_load_model():
    """Download model fastText Wikipedia Indonesia"""
    global model_ready, model_error
    
    model_path = "/tmp/id_model.txt"
    
    # Cek apakah sudah ada
    if os.path.exists(model_path):
        print("Model sudah ada, langsung load...")
        load_model_from_text(model_path)
        return
    
    # Download model Wikipedia Indonesia dari fastText
    # Model ini sudah terlatih dengan Wikipedia Bahasa Indonesia
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.id.vec"
    
    print(f"Downloading model dari {url}...")
    print("Ini mungkin butuh 2-5 menit (file ~600MB)...")
    
    try:
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = block_num * block_size / total_size * 100
                if int(pct) % 10 == 0:
                    print(f"Download: {min(pct, 100):.0f}%")
        
        urllib.request.urlretrieve(url, model_path, progress)
        print("Download selesai! Loading model...")
        load_model_from_text(model_path)
    except Exception as e:
        model_error = str(e)
        print(f"Download gagal: {e}")
        # Gunakan mini model hardcoded sebagai fallback
        use_mini_model()

def use_mini_model():
    """
    Mini model fallback - pakai similarity yang sudah dikalkulasi
    Jauh lebih akurat dari random, cukup untuk testing
    """
    global model_ready, word_vectors, vocab_list
    
    print("Menggunakan mini model fallback...")
    
    # Kelompok kata berdasarkan kedekatan semantik
    # Setiap kelompok punya "pusat" konsep
    groups = {
        "air_kelompok":    ["air","minum","basah","cair","sungai","laut","hujan","danau","kolam","embun","es","banjir","sumur","jernih","tetes","mengalir","sejuk","segar","mineral"],
        "api_kelompok":    ["api","panas","bara","nyala","membakar","asap","abu","kebakaran","korek","lilin","obor","kompor","kayu","arang","merah","hangat","cahaya","terang"],
        "langit_kelompok": ["langit","biru","awan","tinggi","luas","matahari","bulan","bintang","mendung","cerah","hujan","petir","pelangi","senja","fajar","cakrawala","terbang","burung"],
        "laut_kelompok":   ["laut","ombak","pantai","ikan","kapal","asin","dalam","biru","samudra","nelayan","jaring","penyu","paus","hiu","cumi","udang","kerang","pasir","pulau"],
        "hutan_kelompok":  ["hutan","pohon","lebat","rimba","daun","ranting","akar","batang","hijau","binatang","satwa","harimau","gajah","monyet","burung","jamur","lumut","kabut"],
        "makanan_kelompok":["nasi","makan","lauk","masak","kenyang","piring","sendok","warung","restoran","dapur","bumbu","pedas","manis","gurih","lapar","minum","enak","lezat"],
        "manusia_kelompok":["manusia","orang","tubuh","tangan","kaki","kepala","mata","mulut","hidung","telinga","rambut","jantung","darah","tulang","kulit","hati","jiwa","pikiran"],
        "kota_kelompok":   ["kota","jalan","gedung","mobil","motor","macet","ramai","penduduk","pasar","toko","mall","kantor","sekolah","rumah","jembatan","lampu","polisi","bus"],
        "alam_kelompok":   ["alam","gunung","bukit","lembah","sungai","danau","hutan","pantai","padang","sawah","ladang","kebun","desa","udara","bersih","hijau","sejuk","tenang"],
        "perasaan_kelompok":["hati","cinta","rindu","sedih","gembira","takut","marah","bahagia","senang","sakit","lelah","harapan","mimpi","kenangan","rasa","jiwa","emosi","perasaan"],
    }
    
    # Buat vektor sederhana berdasarkan kelompok
    dim = len(groups)
    group_names = list(groups.keys())
    
    all_words = {}
    for i, (group, words) in enumerate(groups.items()):
        for j, word in enumerate(words):
            vec = np.zeros(dim, dtype=np.float32)
            # Kata dalam kelompok yang sama punya nilai tinggi di dimensi kelompoknya
            # Makin dekat ke awal list = makin tinggi nilainya
            vec[i] = 1.0 - (j * 0.03)
            # Sedikit noise untuk variasi
            for k in range(dim):
                if k != i:
                    vec[k] = np.random.uniform(0, 0.1)
            all_words[word] = vec
    
    word_vectors.update(all_words)
    vocab_list.extend(list(all_words.keys()))
    model_ready = True
    print(f"Mini model ready: {len(word_vectors)} kata")

# Start download di background thread
threading.Thread(target=download_and_load_model, daemon=True).start()

# ============================================================
# ENDPOINT: Generate ranking untuk kata rahasia
# ============================================================
@app.route("/generate-ranking", methods=["POST"])
def generate_ranking():
    if request.headers.get("x-roblox-secret") != ROBLOX_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    
    if not model_ready:
        if model_error:
            return jsonify({"error": f"Model error: {model_error}"}), 500
        return jsonify({"error": "Model masih loading, tunggu sebentar..."}), 503
    
    data = request.get_json()
    kata_rahasia = (data.get("kata_rahasia") or "").lower().strip()
    
    if not kata_rahasia:
        return jsonify({"error": "kata_rahasia diperlukan"}), 400
    
    if kata_rahasia not in word_vectors:
        return jsonify({"error": f"'{kata_rahasia}' tidak ada di model"}), 404
    
    target_vec = word_vectors[kata_rahasia]
    
    # Hitung similarity ke semua kata
    # Filter hanya kata Indonesia: ada vokal, panjang wajar, semua huruf
    def is_valid_indonesian(w):
        if len(w) < 2 or len(w) > 18:
            return False
        if not w.isalpha():
            return False
        # Harus ada minimal 1 vokal
        if not any(c in 'aiueo' for c in w):
            return False
        # Tidak boleh ada 4+ konsonan berurutan (bukan kata Indonesia)
        consonants = set('bcdfghjklmnpqrstvwxyz')
        streak = 0
        for c in w:
            if c in consonants:
                streak += 1
                if streak >= 4:
                    return False
            else:
                streak = 0
        # Hindari kata yang terlalu banyak huruf langka (x, q, z berlebihan)
        rare = sum(1 for c in w if c in 'xqz')
        if rare > 1:
            return False
        return True
    
    similarities = []
    for word, vec in word_vectors.items():
        if word == kata_rahasia:
            continue
        if not is_valid_indonesian(word):
            continue
        sim = float(cosine_similarity(target_vec, vec))
        similarities.append((word, sim))
    
    # Sort dari paling mirip ke paling jauh
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Ambil top 15000 untuk jangkauan luas tapi tidak timeout
    TOP_N = 15000
    similarities = similarities[:TOP_N]
    
    # Convert ke ranking
    ranking = {}
    ranking[kata_rahasia] = 1
    for i, (word, sim) in enumerate(similarities):
        ranking[word] = i + 2
    
    total = len(ranking)
    print(f"[RANKING] '{kata_rahasia}' -> {total} kata valid Indonesia (word2vec)")
    
    return jsonify({
        "success": True,
        "kata_rahasia": kata_rahasia,
        "ranking": ranking,
        "jumlah": total,
        "method": "word2vec"
    })


# ============================================================
# ENDPOINT: Cek apakah kata ada di model (filter bahasa Indonesia)
# ============================================================
@app.route("/check-word", methods=["POST"])
def check_word():
    if request.headers.get("x-roblox-secret") != ROBLOX_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    
    if not model_ready:
        return jsonify({"valid": True})  # kalau model belum siap, loloskan saja
    
    data = request.get_json()
    kata = (data.get("kata") or "").lower().strip()
    
    if not kata:
        return jsonify({"valid": False, "alasan": "Kata kosong"})
    
    ada_di_model = kata in word_vectors
    
    return jsonify({
        "valid": ada_di_model,
        "kata": kata,
        "alasan": None if ada_di_model else "Bukan kata bahasa Indonesia yang dikenal"
    })

# ============================================================
# ENDPOINT: Generate kata rahasia random
# ============================================================
@app.route("/generate-word", methods=["POST"])
def generate_word():
    if request.headers.get("x-roblox-secret") != ROBLOX_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    
    if not model_ready:
        return jsonify({"error": "Model masih loading"}), 503
    
    import random
    
    # Filter dari KATA_LAYAK: harus ada di model dan belum dipakai
    available = [k for k in KATA_LAYAK 
                 if k in word_vectors and k not in used_words]
    
    if not available:
        # Reset dan coba lagi
        used_words.clear()
        available = [k for k in KATA_LAYAK if k in word_vectors]
    
    if available:
        kata = random.choice(available)
    else:
        # Fallback keras: ambil dari vocab model dengan filter ketat
        candidates = [w for w in vocab_list 
                     if (4 <= len(w) <= 12 
                         and w.isalpha() 
                         and w[0] in 'abcdefghijklmnoprstuw'  # huruf awal umum Indonesia
                         and not any(c in w for c in 'qvxyz')  # hindari huruf jarang
                         and w not in used_words)]
        kata = random.choice(candidates) if candidates else "laut"
    
    used_words.add(kata)
    print(f"[WORD] Terpilih: '{kata}' (sisa: {len(available)-1} kata layak)")
    
    return jsonify({"success": True, "kata": kata.upper()})

# ============================================================
# Health check
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_ready": model_ready,
        "model_error": model_error,
        "vocab_size": len(word_vectors),
        "used_words": len(used_words),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
