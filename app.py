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

# Cache ranking di level proxy (antar semua Roblox server)
# { kata_rahasia: {ranking_dict} }
proxy_ranking_cache = {}
cache_lock = threading.Lock()

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

# ============================================================
# SEMANTIC CLUSTERS - Boost kedekatan kata berdasarkan pengetahuan manusia
# Mengatasi kelemahan model Word2Vec yang dilatih dari Wikipedia formal
# Format: { kata_pusat: [kata_sangat_dekat, ...] }
# ============================================================
SEMANTIC_CLUSTERS = {
    # ALAM & LINGKUNGAN
    "hutan":    ["pohon","rimba","belantara","lebat","semak","belukar","daun","ranting","akar","batang","lumut","jamur","bambu","rotan","kayu","satwa","harimau","gajah","orangutan","monyet","rusa","burung","deforestasi","konservasi","tropis","lembab","hijau","teduh","rindang","cagar","margasatwa","ekosistem"],
    "laut":     ["ikan","ombak","pantai","samudra","nelayan","kapal","perahu","jaring","terumbu","karang","lumba","paus","hiu","cumi","udang","kerang","garam","asin","pasir","pelabuhan","biru","gelombang","pesisir","teluk","selat","pulau","tsunami","lautnya","lautan","bahari","maritim"],
    "gunung":   ["puncak","lereng","kawah","vulkanik","mendaki","pendaki","trekking","hutan","kabut","dingin","salju","lava","magma","meletus","erupsi","gunung berapi","tebing","jurang","batu","kerikil","savana","padang","edelweiss","jalur","camp","basecamp","summit","altitude"],
    "sungai":   ["air","mengalir","arus","hulu","hilir","muara","jembatan","perahu","ikan","banjir","bendungan","irigasi","sawah","tepi","bantaran","keruh","jernih","deras","tenang","danau","rawa","delta","erosi","sedimen"],
    "api":      ["nyala","bara","panas","membakar","asap","abu","arang","korek","lilin","obor","kompor","kayu","kebakaran","hangus","gosong","cahaya","terang","unggun","merah","oranye","lidah api","percikan","tungku","perapian","damkar","pemadam"],
    "air":      ["minum","basah","cair","sungai","danau","hujan","kolam","sumur","embun","es","banjir","jernih","mengalir","sejuk","segar","mineral","pompa","pipa","waduk","bendungan","mata air","tetes","genangan","kelembaban","hidrogen","oksigen"],
    "angin":    ["hembus","tiup","badai","topan","kencang","sepoi","sejuk","tornado","siklon","ribut","puting beliung","kecepatan","arah","barat","timur","utara","selatan","monsun","angin laut","angin darat","layar","layang","dingin","segar"],
    "hujan":    ["tetes","lebat","gerimis","deras","banjir","petir","kilat","mendung","awan","basah","payung","jas hujan","musim hujan","cuaca","pelangi","segar","tanah","genangan","selokan","banjir","kabut"],

    # KERAJAAN & SEJARAH
    "raja":     ["ratu","mahkota","kerajaan","istana","pangeran","putri","tahta","singgasana","permaisuri","bangsawan","dinasti","sultan","kaisar","khalifah","pemimpin","penguasa","rakyat","kerajaan","perang","tentara","pahlawan","pedang","benteng","prajurit","bala","wilayah","kekuasaan","takhta","wangsa","adipati","adipati","bangsawan","ningrat","harem","keputren","penakluk","penjajah","menaklukkan","memerintah","bertahta","mahkamah","pengadilan","hukum","pajak","upeti","patih","mahapatih","senopati","panglima"],
    "kerajaan": ["raja","ratu","istana","mahkota","tahta","bangsawan","pangeran","putri","dinasti","sultan","kaisar","rakyat","perang","tentara","pedang","prajurit","wilayah","kekuasaan","kejayaan","runtuh","takluk","jajah","koloni","imperium","monarki","kerajaan kuno","majapahit","sriwijaya","mataram","pajang","demak"],
    "istana":   ["raja","ratu","kerajaan","mahkota","pangeran","putri","bangsawan","megah","mewah","bangunan","taman","penjaga","pengawal","singgasana","aula","balairung","dapur","kamar","menara","benteng","tembok","parit","gerbang","upacara","pesta","jamuan"],
    "perang":   ["tentara","prajurit","senjata","pedang","panah","tombak","senapan","bom","peluru","musuh","pertempuran","medan","menyerang","bertahan","kalah","menang","korban","pahlawan","jenderal","komandan","strategi","taktik","benteng","pertahanan","serangan","invasi","penaklukan","gencatan senjata","perdamaian","perjanjian"],
    "pahlawan": ["hero","pejuang","pemberani","berani","berjuang","berkorban","bangsa","negara","kemerdekaan","melawan","penjajah","tentara","perang","medal","penghargaan","legenda","sejarah","teladan","inspirasi","patriot","nasionalis"],

    # PENDIDIKAN
    "sekolah":  ["kelas","guru","murid","siswa","buku","pelajaran","tugas","ujian","nilai","rapor","les","belajar","perpustakaan","kantin","lapangan","seragam","tas","pensil","pulpen","bangku","kurikulum","ulangan","pr","wisuda","ijazah","sd","smp","sma","universitas","kampus","mahasiswa","dosen","kuliah","skripsi","semester"],
    "buku":     ["baca","halaman","cerita","penulis","pengarang","penerbit","novel","kamus","ensiklopedia","komik","majalah","koran","perpustakaan","toko buku","sampul","judul","bab","paragraf","kalimat","kata","ilmu","pengetahuan","fiksi","nonfiksi","puisi","sastra"],
    "guru":     ["murid","siswa","kelas","sekolah","mengajar","pelajaran","ilmu","papan tulis","nilai","ujian","rapor","pendidikan","profesional","honorer","sertifikasi","tunjangan","pengajar","mentor","pembimbing","wali kelas","kepala sekolah","dosen","lecturer"],

    # MAKANAN & MINUMAN
    "makanan":  ["nasi","lauk","sayur","kenyang","lapar","enak","lezat","pedas","manis","asin","gurih","warung","restoran","piring","sendok","garpu","jajanan","kue","roti","sate","bakso","mie","sup","camilan","gizi","nutrisi","kalori","menu","masak","resep"],
    "nasi":     ["beras","piring","lauk","sayur","makan","kenyang","pulen","liwet","uduk","goreng","putih","kukus","lontong","ketupat","bubur","tim","gabah","sawah","petani","padi","penggilingan"],
    "masak":    ["bumbu","rempah","dapur","kompor","wajan","panci","pisau","minyak","garam","gula","bawang","cabai","kunyit","jahe","santan","kecap","resep","goreng","rebus","kukus","bakar","tumis","tepung","adonan","matang","mentah","racik"],
    "kopi":     ["minum","pahit","manis","susu","gula","cangkir","cafe","barista","espresso","cappuccino","latte","robusta","arabika","biji","panen","kebun","roasting","grinding","aroma","hangat","pagi","kafe","ngopi"],
    "gula":     ["manis","tebu","kelapa","aren","pasir","halus","merah","putih","kue","minuman","diabetes","kalori","sirup","madu","pemanis","kristal","larut","karamel","coklat"],

    # TRANSPORTASI
    "kapal":    ["laut","berlayar","pelabuhan","nelayan","awak","kapten","penumpang","kargo","jangkar","layar","mesin","lambung","dek","kemudi","mercusuar","badai","ombak","samudra","ferry","kapal selam","kapal perang","ekspedisi"],
    "pesawat":  ["terbang","pilot","bandara","penumpang","tiket","landasan","sayap","mesin","jet","ketinggian","awan","turbulens","pramugari","bagasi","kargo","penerbangan","rute","internasional","domestik","boarding","terminal"],
    "mobil":    ["jalan","mengemudi","sopir","bensin","solar","mesin","roda","kemudi","rem","gas","klakson","spion","kaca","ban","garasi","parkir","kemacetan","lalu lintas","polisi","sim","stnk","dealer","merk"],

    # PROFESI
    "dokter":   ["pasien","sakit","obat","rumah sakit","klinik","periksa","diagnosa","resep","operasi","perawat","bidan","apoteker","spesialis","penyakit","sembuh","sehat","medis","kesehatan","stetoskop","jarum suntik","infus"],
    "polisi":   ["hukum","kejahatan","pelaku","korban","laporan","penyelidikan","penyidikan","patroli","senjata","borgol","penjara","jaksa","hakim","pengadilan","tilang","lalu lintas","keamanan","ketertiban","masyarakat"],

    # RUMAH & BANGUNAN
    "rumah":    ["kamar","dapur","ruang tamu","kamar mandi","garasi","teras","halaman","pagar","pintu","jendela","atap","lantai","dinding","tangga","sofa","meja","kursi","lemari","kasur","bantal","lampu","listrik","kontrakan","kos","sewa","bangunan","tembok","pondasi"],
    "dapur":    ["masak","kompor","panci","wajan","pisau","bumbu","bahan","minyak","kulkas","rak","piring","gelas","sendok","garpu","talenan","wastafel","air","sabun","cuci","bersih","asap","aroma","resep"],

    # ALAM SEMESTA
    "bulan":    ["malam","bintang","langit","sinar","gelap","terang","gerhana","purnama","sabit","gravitasi","pasang","surut","astronot","luar angkasa","orbit","bumi","matahari","bulan baru","romantis","puisi","laut"],
    "matahari": ["sinar","panas","cahaya","terang","terbit","terbenam","fajar","senja","bumi","bulan","bintang","planet","tata surya","energi","surya","panel surya","ultraviolet","vitamin d","fotosintesis","hangat","musim"],
    "bintang":  ["langit","malam","kelap kelip","sinar","jauh","galaksi","konstelasi","rasi","astronomi","teleskop","luar angkasa","planet","meteor","supernova","bima sakti","zodiak","ramalan","artis","terkenal","populer","idola"],

    # HEWAN
    "harimau":  ["singa","macan","tutul","belang","buas","liar","predator","hutan","berburu","mangsa","cakar","taring","mengaum","lari","kuat","ganas","terancam","punah","konservasi","kebun binatang","sumatera","kalimantan"],
    "ikan":     ["laut","sungai","kolam","berenang","sirip","insang","sisik","nelayan","memancing","pancing","jaring","segar","asin","goreng","bakar","kuah","protein","terumbu karang","aquarium","budidaya","tambak"],
    "burung":   ["terbang","sayap","bulu","paruh","sarang","telur","langit","pohon","berkicau","suara","camar","elang","merpati","nuri","beo","kakatua","merak","bangau","flamingo","migrasi","bebas"],

    # OLAHRAGA & HIBURAN
    "olahraga": ["sepak bola","basket","voli","badminton","renang","lari","gym","latihan","pertandingan","kompetisi","juara","medali","atlet","pemain","pelatih","stadion","lapangan","bola","raket","gol","skor","tim","klub","sehat","kebugaran"],
    "musik":    ["lagu","nada","melodi","ritme","irama","suara","vokal","penyanyi","band","gitar","piano","drum","bass","biola","keyboard","studio","rekaman","konser","album","hits","genre","jazz","pop","rock","lirik","chord"],

    # TEKNOLOGI
    "teknologi":["komputer","laptop","hp","smartphone","internet","wifi","aplikasi","software","hardware","program","coding","data","server","cloud","ai","robot","digital","online","website","medsos","youtube","streaming","inovasi","canggih"],
    "internet": ["online","website","browsing","wifi","data","email","media sosial","youtube","google","download","upload","streaming","jaringan","koneksi","server","bandwidth","viral","konten","kreator","influencer"],

    # ALAM & CUACA
    "hujan":    ["tetes","lebat","gerimis","deras","banjir","petir","kilat","mendung","awan","basah","payung","jas hujan","musim","cuaca","pelangi","segar","tanah","genangan","banjir","kabut","dingin","sejuk"],
    "angin":    ["hembus","tiup","badai","topan","kencang","sepoi","sejuk","tornado","siklon","ribut","dingin","segar","layar","layang","kecepatan","arah","barat","timur","utara","selatan","monsun"],

    # PERTANIAN
    "sawah":    ["padi","beras","nasi","petani","tanam","panen","irigasi","air","lumpur","traktor","cangkul","bajak","gabah","jerami","ladang","kebun","desa","hijau","subur","pupuk","pestisida"],
    "petani":   ["sawah","ladang","kebun","panen","tanam","bibit","pupuk","traktor","cangkul","beras","jagung","sayur","buah","sapi","kambing","desa","sederhana","kerja keras","hasil bumi","musim"],

    # KELUARGA & SOSIAL
    "keluarga": ["ayah","ibu","anak","kakak","adik","kakek","nenek","paman","bibi","sepupu","saudara","suami","istri","orang tua","menikah","rumah tangga","keturunan","cucu","mertua","ipar","harmonis","hangat"],
    "cinta":    ["kasih","sayang","rindu","kangen","romansa","pasangan","kekasih","suami","istri","menikah","romantis","bunga","coklat","ciuman","pelukan","setia","patah hati","putus","jodoh","jatuh cinta","perasaan"],

    # KESEHATAN
    "kesehatan":["dokter","obat","sakit","rumah sakit","puskesmas","perawat","apotek","vaksin","imun","demam","batuk","flu","luka","operasi","rawat","sembuh","sehat","gizi","nutrisi","diet","olahraga","istirahat"],

    # PENDIDIKAN LANJUT
    "ilmu":     ["pengetahuan","belajar","sekolah","buku","guru","murid","sains","teknologi","riset","penelitian","percobaan","teori","fakta","data","analisis","kesimpulan","cerdas","pandai","pintar","akademik"],

    # KOTA & INFRASTRUKTUR
    "kota":     ["jalan","gedung","mobil","motor","macet","ramai","penduduk","pasar","toko","mall","kantor","sekolah","jembatan","lampu","polisi","bus","angkot","taksi","trotoar","taman","plaza","metropolitan","urban"],
}

def get_semantic_boost(kata_rahasia, kata_tebak):
    """
    Kembalikan boost ranking berdasarkan semantic cluster.
    Makin tinggi boost = makin dekat ke target.
    0 = tidak ada relasi cluster
    """
    kata_l = kata_rahasia.lower()
    tebak_l = kata_tebak.lower()
    
    # Cek apakah kata rahasia ada di cluster
    if kata_l in SEMANTIC_CLUSTERS:
        related = SEMANTIC_CLUSTERS[kata_l]
        # Cek posisi kata tebak di list (makin depan = makin dekat)
        for i, r in enumerate(related):
            if r == tebak_l:
                # Posisi 0-5: boost sangat tinggi (ranking ~5-20)
                # Posisi 6-15: boost tinggi (ranking ~20-100)
                # Posisi 16+: boost sedang (ranking ~100-300)
                if i < 6:
                    return 10 - i  # 10, 9, 8, 7, 6, 5
                elif i < 16:
                    return 4
                else:
                    return 2
    return 0

def cosine_similarity(v1, v2):
    """Hitung cosine similarity antara 2 vektor"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    return dot / norm

def load_model_from_text(filepath):
    """
    Load model fastText - hanya 50K kata teratas.
    50K kata x 300 dim x 4 bytes = ~60MB RAM (aman di Railway free tier 512MB).
    Kata teratas di file = paling sering di Wikipedia = kata umum pasti ada.
    """
    global word_vectors, vocab_list, model_ready, model_error
    MAX_WORDS = 50000
    try:
        print(f"Loading model (max {MAX_WORDS} kata)...")
        count = 0
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            parts_h = header.strip().split()
            dims = int(parts_h[1]) if len(parts_h) >= 2 else 300
            for line in f:
                if count >= MAX_WORDS:
                    break
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
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
        # Cek kata penting
        test = ["hutan","pohon","laut","api","air","gunung","rumah","nasi"]
        ada = [w for w in test if w in word_vectors]
        tidak = [w for w in test if w not in word_vectors]
        print(f"Kata kunci ada di model: {ada}")
        if tidak:
            print(f"TIDAK ADA di model: {tidak}")
        model_ready = True
        # Filter KATA_LAYAK: hanya yang ada di model
        global KATA_LAYAK
        KATA_LAYAK = [k for k in KATA_LAYAK if k in word_vectors]
        print(f"KATA_LAYAK valid: {len(KATA_LAYAK)} kata")
    except Exception as e:
        model_error = str(e)
        print(f"Error loading model: {e}")

def download_and_load_model():
    """Download model fastText Wikipedia Indonesia"""
    global model_ready, model_error
    
    model_path = "/tmp/id_model.txt"
    
    # Hapus model lama jika ada (agar pakai versi terbaru dengan limit 50K)
    # Uncomment baris di bawah jika perlu force re-download:
    # if os.path.exists(model_path): os.remove(model_path)
    
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
    
    # Cek cache proxy dulu (hemat waktu & CPU)
    with cache_lock:
        if kata_rahasia in proxy_ranking_cache:
            cached = proxy_ranking_cache[kata_rahasia]
            print(f"[RANKING] Cache hit: '{kata_rahasia}'")
            return jsonify({"success": True, "kata_rahasia": kata_rahasia,
                           "ranking": cached, "jumlah": len(cached), "method": "cache"})
    
    target_vec = word_vectors[kata_rahasia]
    
    # Hitung similarity menggunakan numpy vectorized (jauh lebih cepat dari loop)
    def is_valid_indonesian(w):
        if len(w) < 2 or len(w) > 18:
            return False
        if not w.isalpha():
            return False
        if not any(c in 'aiueo' for c in w):
            return False
        consonants = set('bcdfghjklmnpqrstvwxyz')
        streak = 0
        for c in w:
            if c in consonants:
                streak += 1
                if streak >= 4:
                    return False
            else:
                streak = 0
        rare = sum(1 for c in w if c in 'xqz')
        if rare > 1:
            return False
        return True
    
    # Filter kata valid dulu
    valid_words = [(w, v) for w, v in word_vectors.items() 
                   if w != kata_rahasia and is_valid_indonesian(w)]
    
    # Numpy vectorized cosine similarity (10-20x lebih cepat dari loop biasa)
    if valid_words:
        words_list = [w for w, v in valid_words]
        vecs_matrix = np.array([v for w, v in valid_words], dtype=np.float32)
        
        # Normalisasi target
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-10)
        
        # Normalisasi semua vektor sekaligus
        norms = np.linalg.norm(vecs_matrix, axis=1, keepdims=True) + 1e-10
        vecs_normalized = vecs_matrix / norms
        
        # Dot product sekaligus (vectorized)
        sims = vecs_normalized.dot(target_norm).tolist()
        
        similarities = list(zip(words_list, sims))
    else:
        similarities = []
    
    # Sort dan ambil top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    TOP_N = 15000
    similarities = similarities[:TOP_N]
    
    # Convert ke ranking dasar dari Word2Vec
    wv_ranking = {kata_rahasia: 1}
    for i, (word, sim) in enumerate(similarities):
        wv_ranking[word] = i + 2
    
    # Terapkan Semantic Boost (hibrid Word2Vec + cluster pengetahuan)
    # Boost menggeser kata-kata yang secara semantik jelas berkaitan
    ranking = {}
    kata_l = kata_rahasia.lower()
    
    # Kelompok kata yang mendapat boost berdasarkan cluster
    boosted = {}
    if kata_l in SEMANTIC_CLUSTERS:
        related = SEMANTIC_CLUSTERS[kata_l]
        for i, r in enumerate(related):
            if r in wv_ranking:
                if i < 6:
                    boosted[r] = i + 2       # ranking 2-7 (sangat dekat)
                elif i < 16:
                    boosted[r] = i + 8       # ranking ~10-24 (dekat)
                else:
                    boosted[r] = i + 20      # ranking ~36-70 (cukup dekat)
    
    # Susun ranking final:
    # 1. Kata rahasia sendiri selalu #1
    # 2. Kata yang di-boost masuk ke posisi awal
    # 3. Sisanya dari Word2Vec, digeser setelah yang di-boost
    
    boosted_set = set(boosted.keys())
    
    # Hitung berapa slot yang "dipakai" oleh boosted words
    # Word2Vec words lain digeser ke bawah
    boost_count = len(boosted)
    
    ranking[kata_rahasia] = 1
    for word, brank in boosted.items():
        ranking[word] = brank
    
    # Sisanya dari Word2Vec similarity (skip yang sudah di-boost)
    current_rank = boost_count + 2  # mulai setelah boosted words
    for word, sim in similarities:
        if word == kata_rahasia or word in boosted_set:
            continue
        ranking[word] = current_rank
        current_rank += 1
    
    # Tambah kata-kata cluster yang tidak ada di Word2Vec (dengan ranking tinggi)
    for i, r in enumerate(SEMANTIC_CLUSTERS.get(kata_l, [])):
        if r not in ranking:
            if i < 6:
                ranking[r] = i + 2
            elif i < 16:
                ranking[r] = i + 8
            else:
                ranking[r] = i + 20
    
    total = len(ranking)
    boosted_info = list(boosted.items())[:5] if boosted else []
    print(f"[RANKING] '{kata_rahasia}' -> {total} kata | boost: {len(boosted)} kata | contoh: {boosted_info}")
    
    # Simpan ke proxy cache (max 50 kata untuk hemat memory)
    with cache_lock:
        if len(proxy_ranking_cache) >= 50:
            # Hapus entry paling lama
            oldest = next(iter(proxy_ranking_cache))
            del proxy_ranking_cache[oldest]
        proxy_ranking_cache[kata_rahasia] = ranking
    
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
