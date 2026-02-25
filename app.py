"""
KONTEKS Proxy v8
Word2Vec fastText + PRECOMPUTED_RELATIONS manual super lengkap
Mencakup semua kata di KATA_LAYAK dengan relasi yang akurat
"""

from flask import Flask, request, jsonify
import os, numpy as np, threading, time, gzip, struct

app = Flask(__name__)
ROBLOX_SECRET = os.environ.get("ROBLOX_SECRET", "konteks-rahasia-2024")

proxy_ranking_cache = {}
cache_lock = threading.Lock()
word_vectors = {}
vocab_list   = []
model_ready  = False
model_error  = None

# ============================================================
# KATA RAHASIA
# ============================================================
KATA_LAYAK = sorted(set([
    # ALAM
    "air","api","angin","awan","batu","bulan","bunga","burung","cahaya","daun",
    "embun","fajar","galaksi","gunung","hutan","hujan","ikan","kabut","kilat",
    "langit","laut","malam","matahari","pantai","pasir","pelangi","petir","pohon",
    "pulau","salju","sungai","tanah","udara","bintang","badai","danau","lembah",
    # HEWAN
    "harimau","singa","gajah","monyet","rusa","kuda","sapi","kambing","ayam",
    "bebek","ular","buaya","katak","tikus","kelinci","kucing","anjing","rubah",
    "beruang","panda","zebra","jerapah","lumba","paus","hiu","elang","lebah",
    "kupu","semut","lele","udang","kepiting","penyu","kumbang","capung",
    # TUMBUHAN
    "mawar","melati","anggrek","bambu","jati","pinus","cemara","beringin",
    "mangga","rambutan","durian","pisang","pepaya","kelapa","padi","jagung",
    "singkong","kentang","wortel","bayam","cabai","tomat","bawang","jahe",
    "kunyit","kopi","teh","salak","manggis","alpukat","nanas","semangka","jamur",
    # MAKANAN
    "nasi","sate","bakso","soto","rendang","tempe","tahu","mie","roti","kue",
    "susu","gula","garam","madu","sambal","kecap","telur","daging",
    # PROFESI
    "dokter","guru","polisi","tentara","petani","nelayan","pedagang","pengacara",
    "hakim","arsitek","pilot","sopir","koki","penulis","seniman","musisi","atlet",
    # TUBUH
    "kepala","mata","hidung","mulut","telinga","tangan","kaki","jantung","paru",
    "otak","tulang","kulit","darah","napas","rambut","perut","dada",
    # PERASAAN
    "cinta","rindu","sedih","bahagia","marah","takut","bangga","malu","senang",
    # TEMPAT
    "rumah","sekolah","masjid","gereja","istana","pasar","kantor","hotel",
    "perpustakaan","stadion","bandara","pelabuhan","jembatan","menara","taman",
    # TRANSPORTASI
    "mobil","motor","kapal","pesawat","kereta","sepeda","perahu","bus","taksi",
    # TEKNOLOGI
    "komputer","internet","aplikasi","robot","kamera","telepon",
    # SENI
    "musik","lagu","tari","wayang","batik","gamelan","film","buku",
    # KERAJAAN
    "raja","ratu","pangeran","putri","sultan","kerajaan","pedang","mahkota","tahta",
    # ALAM SEMESTA
    "matahari","bulan","bintang","planet","bumi","galaksi","meteor","komet",
    # KONSEP
    "mimpi","waktu","uang","ilmu","damai","harapan","kehidupan",
    # INDONESIA
    "sawah","desa","orangutan","komodo","tempe","rendang","wayang","batik",
    # WARNA
    "merah","biru","kuning","hijau","putih","hitam","oranye","ungu",
    # SIFAT
    "panas","dingin","besar","kecil","cepat","lambat","kuat","lemah",
    # OLAHRAGA
    "sepak bola","basket","badminton","renang","lari","tinju","silat",
    # KELUARGA
    "ayah","ibu","kakak","adik","kakek","nenek","suami","istri","anak","teman",
    # PERANG
    "perang","pahlawan","pedang","benteng","prajurit",
]))

used_words = set()

# ============================================================
# PRECOMPUTED_RELATIONS
# Ranking manual untuk SETIAP kata di KATA_LAYAK
# Urutan = urutan kedekatan (index 0 = paling dekat = rank 2)
# Semakin panjang listnya = semakin banyak kata yang di-boost
# ============================================================
PRECOMPUTED_RELATIONS = {

    # ══════════════════════════════════════════════════════════
    # ALAM & LINGKUNGAN
    # ══════════════════════════════════════════════════════════
    "air": ["minum","basah","cair","sungai","danau","laut","hujan","kolam","sumur",
            "embun","es","banjir","jernih","mengalir","tetes","sejuk","segar","bersih",
            "kotor","keruh","pompa","waduk","bendungan","haus","berenang","ikan",
            "nelayan","irigasi","sawah","uap","dalam","dangkal","biru","mineral",
            "kekeringan","gelombang","pasang","surut","air terjun","selat","teluk",
            "hujan","udara","awan","kabut","embun","salju","es","basah","lembab"],

    "api": ["nyala","bara","panas","membakar","asap","abu","arang","korek","lilin",
            "obor","kompor","kayu","kebakaran","hangus","gosong","cahaya","terang",
            "unggun","merah","oranye","percikan","tungku","perapian","damkar",
            "menyala","berkobar","padam","energi","lahar","vulkanik","matahari",
            "api unggun","pembakaran","sumber energi","udara","oksigen","bahan bakar",
            "bensin","gas","listrik","petir","kilat","panas","api kecil","lilin",
            "obor","senter","lampu","bersinar","terang benderang","hangat","panas"],

    "angin": ["hembus","tiup","sepoi","kencang","udara","segar","dingin","badai",
              "topan","ribut","tornado","langit","awan","layar","layang","nafas",
              "napas","pohon","bergoyang","semilir","laut","darat","siklon","beliung",
              "barat","timur","utara","selatan","monsun","puting beliung","kencang",
              "bertiup","embus","semilir","kesejukan","angin laut","angin darat",
              "angin kencang","badai","topan","siklon","tornado","puting beliung",
              "udara bergerak","oksigen","nafas","pernapasan","paru","hembusan"],

    "awan": ["langit","mendung","hujan","putih","abu","tinggi","terapung","ringan",
             "matahari","petir","angin","uap","air","cuaca","berawan","guntur",
             "mega","senja","fajar","kabut","embun","pelangi","pesawat","terbang",
             "cumulus","nimbus","stratus","gelap","tebal","tipis","hitam","melayang",
             "terbang tinggi","langit mendung","mendung gelap","hujan lebat",
             "kilat","petir","guruh","guntur","hujan deras","mendung tebal"],

    "batu": ["keras","besar","kecil","sungai","gunung","tebing","pasir","tanah",
             "mineral","tambang","bata","semen","bangunan","kristal","permata",
             "karang","laut","kerikil","granit","marmer","gamping","fosil","alam",
             "berat","dingin","batu kali","batu gunung","batuan","bebatuan",
             "batu bata","batu pasir","batu kapur","batu hitam","batu putih",
             "batu cincin","permata","berlian","emas","perak","logam","mineral"],

    "bulan": ["malam","bintang","langit","sinar","gelap","terang","gerhana","purnama",
              "sabit","gravitasi","pasang","surut","astronot","luar angkasa","orbit",
              "bumi","matahari","romantis","cahaya","rembulan","bersinar","indah",
              "bulan baru","bulan purnama","bulan sabit","bulan separuh","gerhana bulan",
              "tata surya","planet","galaksi","bima sakti","satelit","bulan bumi",
              "cahaya bulan","sinar bulan","terang bulan","malam bulan purnama",
              "pasang surut","laut","ombak","gelombang","air laut"],

    "bunga": ["mekar","harum","indah","warna","kelopak","mahkota","putik","serbuk",
              "taman","mawar","melati","anggrek","matahari","lavender","cantik",
              "lebah","kupu","kado","buket","wangi","merah","kuning","putih","ungu",
              "alam","hias","tanaman","pohon","tumbuhan","berkembang","merekah",
              "harum bunga","taman bunga","kebun bunga","bouquet","karangan bunga",
              "bunga segar","bunga layu","kelopak bunga","mahkota bunga","nektar",
              "serbuk sari","lebah madu","kupu-kupu","taman","hijau","daun"],

    "burung": ["terbang","sayap","bulu","paruh","sarang","telur","langit","pohon",
               "berkicau","suara","camar","elang","merpati","nuri","kakaktua","merak",
               "bangau","migrasi","bebas","pagi","kicau","pipit","kutilang","jalak",
               "cendrawasih","kolibri","beo","parkit","lovebird","kenari","tekukur",
               "burung gereja","burung hantu","burung unta","burung pinguin",
               "burung terbang","migrasi burung","sarang burung","telur burung",
               "kicauan","berkicau","bernyanyi","terbang bebas","langit biru",
               "angin","awan","hutan","pohon","ranting","cabang","dahan"],

    "cahaya": ["sinar","terang","matahari","lampu","lilin","api","kilat","pelangi",
               "bercahaya","bersinar","gelap","bayangan","optik","kilau","gemerlap",
               "bintang","bulan","laser","neon","kristal","prisma","warna","putih",
               "panas","energi","foton","terang benderang","bersinar terang",
               "cahaya matahari","cahaya bulan","cahaya bintang","sinar matahari",
               "sinar bulan","sinar ultraviolet","cahaya lampu","penerangan",
               "gelap gulita","remang-remang","redup","temaram","bersinar","kilau",
               "gemerlap","berkelip","berkerlip","bintang berkelip","cahaya redup"],

    "daun": ["hijau","pohon","ranting","gugur","layu","segar","fotosintesis","klorofil",
             "tumbuhan","hutan","alam","dedaunan","teh","tanaman","daun pisang",
             "daun pandan","herbal","obat","kering","basah","lebar","tipis","kuning",
             "coklat","daun jatuh","dedaunan gugur","musim gugur","daun muda",
             "daun tua","daun hijau","daun kuning","daun kering","daun basah",
             "ranting pohon","batang pohon","akar pohon","buah pohon","bunga pohon",
             "pohon rindang","teduh","naungan","bayangan","segar","hijau royo-royo"],

    "embun": ["pagi","basah","dingin","segar","tetes","rumput","daun","kabut","udara",
              "lembab","subuh","fajar","matahari","menetes","bersih","kristal",
              "bening","sejuk","embun pagi","embun malam","embun beku","titik embun",
              "kelembaban","uap air","kondensasi","butiran air","tetes air",
              "segar pagi","udara pagi","dingin pagi","subuh","salat subuh","fajar",
              "terbit matahari","awal pagi","alam pagi","suasana pagi","udara segar"],

    "fajar": ["pagi","matahari terbit","subuh","senja","langit","merah","jingga",
              "kuning","cahaya","bintang","bulan","hari baru","semburat","tenang",
              "sunyi","dingin","embun","ayam berkokok","timur","cahaya pertama",
              "awal hari","pagi buta","dini hari","fajar menyingsing","terang tanah",
              "langit memerah","langit jingga","warna fajar","indah","elok",
              "sunrise","matahari","ufuk timur","cakrawala","alam","keheningan",
              "kedamaian","kesejukan","segar","baru","harapan","semangat"],

    "galaksi": ["bima sakti","bintang","planet","luar angkasa","semesta","teleskop",
                "nebula","supernova","blackhole","gravitasi","orbit","astronomi",
                "tata surya","matahari","cahaya","tahun cahaya","andromeda","spiral",
                "elips","tak beraturan","gugus bintang","awan bintang","nebula",
                "supernova","lubang hitam","materi gelap","energi gelap","kosmos",
                "alam semesta","ekspansi","big bang","astronot","roket","NASA",
                "teleskop hubble","james webb","bintang neutron","pulsar","quasar"],

    "gunung": ["puncak","lereng","kawah","vulkanik","mendaki","pendaki","hutan",
               "kabut","dingin","salju","lava","erupsi","tebing","jurang","batu",
               "edelweiss","camp","summit","tinggi","megah","sungai","lembah",
               "pemandangan","hutan pinus","gunung berapi","gunung mati","kaldera",
               "crater","magma","lahar","abu vulkanik","letusan","gempa bumi",
               "tektonis","geologi","topografi","ketinggian","di atas awan",
               "basecamp","shelter","jalur pendakian","puncak tertinggi","himalaya",
               "everest","semeru","rinjani","bromo","merapi","krakatau","agung"],

    "hutan": ["pohon","rimba","belantara","lebat","semak","belukar","daun","ranting",
              "akar","batang","lumut","jamur","bambu","rotan","kayu","satwa",
              "harimau","gajah","orangutan","monyet","rusa","burung","deforestasi",
              "konservasi","tropis","lembab","hijau","teduh","rindang","margasatwa",
              "ekosistem","oksigen","hutan tropis","hutan hujan","hutan lebat",
              "hutan lindung","hutan produksi","penebangan","reboisasi","penghijauan",
              "keanekaragaman hayati","biodiversitas","flora","fauna","satwa liar",
              "habitat","predator","mangsa","rantai makanan","jaring makanan"],

    "hujan": ["tetes","gerimis","lebat","deras","banjir","petir","kilat","mendung",
              "awan","basah","payung","dingin","sejuk","tanah","genangan","sungai",
              "air","pelangi","segar","bau tanah","musim hujan","hujan deras",
              "hujan lebat","hujan gerimis","rintik-rintik","guyur","curah hujan",
              "musim hujan","kemarau","kering","basah kuyup","kedinginan","jas hujan",
              "payung","sepatu boot","genangan air","banjir","longsor","petir",
              "guruh","guntur","kilat","langit gelap","awan hitam","mendung tebal"],

    "ikan": ["laut","sungai","kolam","berenang","sirip","insang","sisik","nelayan",
             "memancing","jaring","segar","asin","goreng","bakar","protein",
             "terumbu karang","aquarium","budidaya","tambak","bandeng","lele","nila",
             "mas","gurame","salmon","tuna","kerapu","pancing","kail","umpan",
             "joran","perahu","laut","sungai","kolam","tambak","keramba","budidaya",
             "ikan segar","ikan asin","ikan goreng","ikan bakar","ikan asap",
             "makanan laut","seafood","nelayan","kapal ikan","jaring ikan","pukat"],

    "kabut": ["embun","asap","uap","putih","tebal","gunung","dingin","pagi","lembab",
              "redup","udara","sungai","danau","hutan","misterius","tersembunyi",
              "tipis","menyelimuti","kabut pagi","kabut gunung","kabut laut",
              "kabut tipis","kabut tebal","berkabut","lembah berkabut","mistis",
              "dingin","sejuk","basah","lembab","embun","tetes air","kondensasi",
              "uap air","awan rendah","stratus","nimbostratus","cuaca","iklim",
              "visibilitas rendah","jarak pandang","gelap","remang","temaram"],

    "kilat": ["petir","guntur","hujan","badai","langit","listrik","cahaya","terang",
              "cepat","berbahaya","menyambar","api","awan","gelap","menggelegar",
              "flash","kilap","berkilat","berkilap","bersinar","menyilaukan",
              "cahaya sesaat","cahaya terang","ledakan listrik","disambar petir",
              "pohon terbakar","listrik statis","tegangan tinggi","fenomena alam",
              "cuaca buruk","badai petir","thunderstorm","guntur menggelegar"],

    "langit": ["biru","awan","tinggi","luas","matahari","bulan","bintang","mendung",
               "cerah","pelangi","senja","fajar","cakrawala","terbang","burung",
               "udara","angkasa","horizon","mega","batas","gelap","terang",
               "siang","malam","langit biru","langit cerah","langit mendung",
               "langit malam","langit siang","ufuk","kaki langit","cakrawala",
               "luar angkasa","angkasa raya","bintang","planet","bulan","matahari",
               "awan putih","awan hitam","mendung","hujan","pelangi","senja"],

    "laut": ["ombak","pantai","samudra","nelayan","kapal","perahu","jaring","terumbu",
             "karang","lumba","paus","hiu","cumi","udang","kerang","garam","asin",
             "pasir","pelabuhan","biru","gelombang","pesisir","teluk","selat",
             "pulau","tsunami","dalam","luas","ikan","air","angin","laut dalam",
             "dasar laut","tekanan air","biota laut","ekosistem laut","hutan bakau",
             "mangrove","terumbu karang","padang lamun","plankton","rantai makanan",
             "nelayan tradisional","kapal penangkap ikan","pukat","jaring","pancing"],

    "malam": ["gelap","bintang","bulan","sepi","sunyi","tidur","dingin","nyamuk",
              "jangkrik","cahaya","lampu","senja","subuh","mimpi","misterius",
              "hitam","pekat","dini hari","keheningan","istirahat","malam hari",
              "malam gelap","malam sunyi","malam sepi","malam dingin","malam panjang",
              "tengah malam","larut malam","dini hari","waktu tidur","istirahat",
              "bermimpi","tidur nyenyak","lelah","capek","penat","bintang berkelip",
              "bulan bersinar","cahaya bintang","langit malam","kegelapan"],

    "matahari": ["sinar","panas","cahaya","terang","terbit","terbenam","fajar","senja",
                 "bumi","bulan","bintang","planet","tata surya","energi","surya",
                 "ultraviolet","vitamin d","fotosintesis","hangat","musim","kuning",
                 "api","besar","alam","hidup","panas matahari","sinar matahari",
                 "cahaya matahari","matahari terbit","matahari terbenam","fajar",
                 "senja","matahari pagi","matahari siang","matahari sore","terik",
                 "berjemur","tan","kulit","energi surya","panel surya","PLTS"],

    "pantai": ["laut","ombak","pasir","nelayan","kapal","biru","liburan","wisata",
               "kerang","matahari","sunset","sunrise","angin","garam","basah",
               "ikan","terumbu karang","berenang","berjemur","dermaga","batu karang",
               "pantai berpasir","pantai berbatu","pantai indah","pantai bersih",
               "deburan ombak","air laut","pasir putih","pasir kuning","bintang laut",
               "ubur-ubur","kepiting","kerang","tiram","teripang","bulu babi",
               "snorkeling","diving","selancar","surfing","banana boat","kapal selam"],

    "pasir": ["pantai","sungai","gurun","kerikil","tanah","halus","lembut","kuning",
              "coklat","bangunan","batu","emas","jam pasir","sahara","kristal",
              "silika","bermain","ombak","berpasir","pasir putih","pasir hitam",
              "pasir halus","pasir kasar","padang pasir","gurun pasir","bukit pasir",
              "gumuk pasir","angin","badai pasir","gurun sahara","unta","kaktus",
              "oasis","panas terik","kering","gersang","tandus","tak berair"],

    "pelangi": ["warna","merah","jingga","kuning","hijau","biru","nila","ungu",
                "hujan","matahari","cahaya","langit","indah","cerah","harapan",
                "cantik","prisma","bias cahaya","tujuh warna","setelah hujan",
                "pelangi kembar","pelangi ganda","busur pelangi","arc","warna-warni",
                "spektrum cahaya","pembiasan cahaya","tetes hujan","keindahan",
                "alam","fenomena alam","berkas cahaya","optik","cahaya putih",
                "spektrum","merah-jingga-kuning-hijau-biru-nila-ungu"],

    "petir": ["kilat","guntur","hujan","badai","langit","listrik","berbahaya",
              "menggelegar","menyambar","api","suara","cahaya","awan","gelap",
              "sambaran petir","petir menyambar","pohon disambar","bahaya petir",
              "penangkal petir","grounding","listrik","tegangan","voltase","joule",
              "energi petir","kilat cahaya","guntur menggelega","suara keras",
              "memekakkan","mengejutkan","menakutkan","fenomena listrik","plasma"],

    "pohon": ["daun","ranting","akar","batang","buah","bunga","kayu","hutan","teduh",
              "hijau","tumbuh","subur","jati","beringin","mangga","kelapa","bambu",
              "tumbuhan","alam","oksigen","besar","tua","getah","pohon besar",
              "pohon rindang","pohon tua","pohon muda","pohon buah","pohon hias",
              "pohon pelindung","pepohonan","hutan","kebun","ladang","taman",
              "menanam","menebang","reboisasi","penghijauan","lingkungan hidup"],

    "pulau": ["laut","pantai","pasir","ombak","nelayan","kapal","perahu","wisata",
              "tropis","penduduk","flora","fauna","terumbu karang","jembatan",
              "ferry","kepulauan","nusantara","indonesia","pulau kecil","pulau besar",
              "pulau terpencil","pulau wisata","pulau berpenghuni","pulau tak berpenghuni",
              "jawa","sumatera","kalimantan","sulawesi","papua","bali","lombok",
              "maluku","ntt","ntb","kepulauan riau","bangka belitung"],

    "salju": ["putih","dingin","beku","gunung","musim dingin","es","mencair","salju lebat",
              "salju tipis","snowflake","kepingan salju","badai salju","blizzard",
              "musim salju","salju abadi","gletser","es abadi","kutub","arktik",
              "antartika","beruang kutub","rusa kutub","manusia salju","snowman",
              "seluncur es","ski","snowboard","musim dingin","jaket tebal","selimut",
              "hangatkan diri","perapian","api","coklat panas","suhu di bawah nol"],

    "sungai": ["air","mengalir","arus","hulu","hilir","muara","jembatan","perahu",
               "ikan","banjir","bendungan","irigasi","sawah","tepi","deras","jernih",
               "keruh","delta","erosi","sungai besar","sungai kecil","sungai deras",
               "sungai tenang","sungai jernih","sungai keruh","aliran sungai",
               "tepi sungai","bantaran sungai","banjir sungai","luapan sungai",
               "ikan sungai","perahu sungai","menyeberangi","jembatan","tambatan",
               "nelayan sungai","nila","lele","mas","gurame","ikan air tawar"],

    "tanah": ["bumi","lumpur","pasir","batu","subur","kering","sawah","ladang",
              "galian","pertanian","warna","merah","hitam","coklat","mineral",
              "humus","akar","pohon","cacing","mikroba","tanah subur","tanah gersang",
              "tanah liat","tanah pasir","tanah humus","kesuburan","pertanian",
              "bercocok tanam","berladang","berkebun","petani","sawah","ladang",
              "kebun","huma","tegalan","lahan pertanian","erosi tanah","longsor"],

    "udara": ["angin","napas","oksigen","atmosfer","langit","segar","bersih","hembus",
              "paru","lembab","polusi","asap","debu","kabut","suhu","cuaca","awan",
              "angkasa","terbang","ringan","bebas","nitrogen","gas","uap","dingin",
              "panas","bernapas","menghirup","menghembuskan","udara bersih",
              "udara kotor","polusi udara","kualitas udara","PM2.5","emisi",
              "gas rumah kaca","karbon dioksida","nitrogen oksida","sulfur dioksida",
              "udara segar","kesegaran","napas segar","menghirup udara segar"],

    "bintang": ["langit","malam","bersinar","cahaya","galaksi","tata surya","bulan",
                "matahari","indah","berkelip","terang","angkasa","luar angkasa",
                "bima sakti","konstelasi","rasi bintang","orion","scorpio","leo",
                "bintang jatuh","meteor","shooting star","bintang besar","bintang kecil",
                "bintang redup","bintang terang","bintang mati","supernova","neutron",
                "pulsar","quasar","bintang ganda","bintang tiga","gugus bintang",
                "nebula","materi gelap","galaksi spiral","galaksi elips"],

    "badai": ["angin","kencang","hujan","petir","kilat","gelombang","laut","ombak",
              "topan","siklon","tornado","puting beliung","berbahaya","menerjang",
              "menghantam","merusak","bencana","banjir","longsor","angin ribut",
              "badai petir","badai pasir","badai salju","blizzard","hurricane",
              "typhoon","cyclone","storm","cuaca ekstrem","peringatan dini",
              "evakuasi","pengungsian","kerusakan","bangunan runtuh","pohon tumbang"],

    "danau": ["air","tenang","ikan","nelayan","perahu","indah","jernih","biru",
              "pegunungan","hutan","wisata","berenang","mendayung","mancing",
              "danau kawah","danau vulkanik","danau tektonik","danau buatan",
              "waduk","bendungan","embung","kolam besar","genangan air","rawa",
              "mangrove","hutan bakau","ekosistem air tawar","ikan air tawar",
              "katak","capung","teratai","eceng gondok","rumput air","vegetasi air"],

    "lembah": ["gunung","sungai","hijau","subur","bukit","lembah hijau","lembah dalam",
               "ngarai","jurang","tebing","hutan","padang rumput","pemandangan",
               "lembah sempit","lembah luas","lembah fertile","pertanian","sawah",
               "ladang","kebun","desa","perkampungan","kehidupan","air","sungai",
               "mata air","sumber air","dingin","sejuk","kabut","embun"],

    # ══════════════════════════════════════════════════════════
    # HEWAN
    # ══════════════════════════════════════════════════════════
    "harimau": ["singa","macan","predator","buas","hutan","cakar","taring","loreng",
                "kuning","hitam","berbahaya","liar","kuat","cepat","berburu",
                "mangsa","rusa","babi","monyet","harimau sumatera","harimau benggala",
                "harimau siberia","harimau putih","harimau amur","punah","dilindungi",
                "konservasi","suaka margasatwa","taman nasional","WWF","satwa langka",
                "top predator","apex predator","rantai makanan","ekosistem hutan"],

    "singa": ["harimau","predator","buas","Afrika","sabana","mane","rambut","surai",
              "kelompok","kawanan","berburu","mangsa","zebra","rusa","kijang",
              "singa jantan","singa betina","anak singa","pride","savana","padang",
              "Afrika","kenya","tanzania","serengeti","maasai mara","taman nasional",
              "raja hutan","raja binatang","lambang","mahkota","simbol","heraldik"],

    "gajah": ["besar","belalai","gading","telinga","abu","hutan","Afrika","Asia",
              "gajah asia","gajah Afrika","gajah sumatera","gajah kalimantan",
              "herbivora","daun","rumput","buah","kuat","cerdas","ingatan kuat",
              "mahout","pawang gajah","gajah jinak","gajah liar","konservasi",
              "punah","dilindungi","ivory","perburuan liar","gading","taman nasional"],

    "monyet": ["pohon","hutan","ekor","lompat","pisang","buah","cerdas","sosial",
               "kelompok","kawanan","orangutan","simpanse","gorila","kera","lutung",
               "siamang","beruk","macaque","gibbon","macaque","babon","mandrill",
               "bonobo","primata","mamalia","hewan sosial","hierarki","alfa","beta",
               "hutan tropis","pohon tinggi","bergelantungan","akrobat"],

    "rusa": ["hutan","tanduk","cepat","berlari","herbivora","rumput","daun","kijang",
             "banteng","menjangan","rusa jantan","rusa betina","anak rusa","tanduk rusa",
             "rusa totol","rusa sambar","rusa timor","rusa bawean","hewan jinak",
             "satwa liar","hutan","padang","savana","harimau mangsa","predator",
             "berlari kencang","melompat","lincah","gesit","elah","babi hutan"],

    "kuda": ["berlari","cepat","gagah","berkuda","joki","pacuan","kuda pacu",
             "kuda tunggang","kuda beban","kuda perang","kuda liar","mustang",
             "bronco","poni","kuda nil","zebra","saudara","ekuus","mamalia",
             "herbivora","rumput","jerami","kandang","istal","tapal kuda","sepatu kuda",
             "jockey","pacu kuda","balap kuda","tandu","kereta kuda","kuda-kudaan"],

    "sapi": ["susu","daging","lembu","sapi perah","sapi potong","banteng","kerbau",
             "herbivora","rumput","jerami","kandang","ternak","petani","ladang",
             "peternakan","sapi limosin","sapi bali","sapi ongole","sapi fries",
             "sapi perah","sapi potong","sapi kurban","idul adha","bakso","rendang",
             "steak","burger","daging sapi","susu sapi","keju","mentega","yogurt"],

    "kambing": ["domba","ternak","herbivora","rumput","susu kambing","daging kambing",
                "bau kambing","kambing etawa","kambing kacang","kambing boer",
                "kambing angora","kambing saanen","wol","bulu","kambing betina",
                "kambing jantan","anak kambing","cempe","kandang","peternak",
                "kurban","idul adha","sate kambing","gulai kambing","tongseng"],

    "ayam": ["telur","daging","kokok","bulu","kandang","ternak","ayam kampung",
             "ayam broiler","ayam petelur","ayam pedaging","jago","betina","anak ayam",
             "piyik","anakan","mencari makan","cacing","jagung","bekatul",
             "ayam goreng","ayam bakar","ayam kecap","opor ayam","soto ayam",
             "nasi ayam","bakso ayam","nugget","sate ayam","rendang ayam"],

    "bebek": ["angsa","unggas","kolam","renang","telur bebek","daging bebek",
              "bebek mentok","bebek peking","bebek lokal","bebek hibrida","bersuara",
              "kwek-kwek","wadah air","danau","sungai","rawa","lumpur","cacing",
              "bebek goreng","bebek betutu","nasi bebek","bebek panggang","sate bebek"],

    "ular": ["berbisa","melata","sisik","lidah","berbahaya","ular kobra","ular python",
             "ular boa","ular viper","ular sanca","ular piton","ular hijau","ular hitam",
             "ular derik","bisa ular","racun","mematikan","mangsa","tikus","katak",
             "kadal","ular air","ular pohon","ular tanah","ular laut","sengatan",
             "taring","bisa","antivenom","anti bisa","penangkaran ular","reptil"],

    "buaya": ["reptil","berbahaya","gigi","air","sungai","rawa","besar","kuat",
              "buaya muara","buaya air tawar","buaya saltwater","buaya nil",
              "buaya alligator","buaya gharial","caiman","predator air","rahang",
              "mengintai","menyerang","liar","berjemur","telur buaya","anak buaya",
              "kulit buaya","tas kulit","sepatu kulit","berbahaya","mematikan"],

    "katak": ["amfibi","kolam","sungai","air","hujan","melompat","kodok","hijau",
              "hijau","berbintik","racun","katak pohon","katak sawah","katak hijau",
              "berudu","kecebong","telur katak","metamorfosis","berubah bentuk",
              "katak dart","katak beracun","racun kulit","predator serangga",
              "serangga","nyamuk","lalat","cacing","mangsa","lidah panjang"],

    "tikus": ["rumah","kotor","mencuri","makanan","gudang","selokan","penyakit",
              "wabah","pes","leptospirosis","hama","merusak","mengerat","gigi",
              "ekor","whisker","kumis","nocturnal","malam","tikus got","tikus rumah",
              "tikus sawah","hama pertanian","padi","jagung","mencuri makanan",
              "berbahaya","pembawa penyakit","kucing pemangsa","jebakan tikus"],

    "kelinci": ["lucu","lembut","bulu","kuping","panjang","melompat","wortel","sayuran",
                "herbivora","hewan peliharaan","kandang","kelinci angora","kelinci rex",
                "kelinci holland","kelinci mini","kelinci lokal","anak kelinci",
                "bayi kelinci","hamil","beranak","cepat berkembang biak","lembut","jinak"],

    "kucing": ["anjing","hewan peliharaan","bulu","cakar","mencakar","mengeong",
               "tidur","manja","lincah","kucing kampung","kucing Persia","kucing angora",
               "kucing siam","kucing bengal","kucing maine coon","kucing ragdoll",
               "anak kucing","kitten","mengeong","meow","mendengkur","purring",
               "berburu","tikus","mencuri ikan","memanjat","lompat tinggi"],

    "anjing": ["kucing","hewan peliharaan","setia","jinak","menggonggong","ekor",
               "bulu","berlari","anjing kampung","anjing ras","anjing herder",
               "anjing labrador","anjing golden","anjing beagle","anjing bulldog",
               "anak anjing","puppy","gonggong","woof","bark","mencium","hidung",
               "indera penciuman","menjaga","penjaga rumah","anjing polisi","k9"],

    "rubah": ["licik","cerdik","hutan","berekor","ekor lebat","merah","oranye",
              "serigala","saudara","canidae","nocturnal","berburu","tikus","kelinci",
              "rubah merah","rubah arktik","rubah fennec","rubah abu","cerita",
              "folklor","dongeng","simbol","kelicikan","kecerdikan","pintar",
              "mencuri","ayam","kandang","petani","hama","berbahaya bagi ternak"],

    "beruang": ["besar","kuat","hutan","memanjat","madu","lebah","salmon","ikan",
                "beruang kutub","beruang grizzly","beruang madu","beruang hitam",
                "beruang coklat","beruang panda","beruang sun","hibernasi","musim dingin",
                "gua","tidur panjang","ibu beruang","anak beruang","cub","pawang",
                "taman nasional","dilindungi","berbahaya","cakar","taring"],

    "panda": ["hitam putih","China","bambu","lucu","langka","dilindungi","gemuk",
              "panda besar","giant panda","panda merah","red panda","WWF","simbol",
              "konservasi","Chengdu","kebun binatang","bayi panda","anak panda",
              "makan bambu","tidur","malas","jinak","internasional","terkenal"],

    "zebra": ["loreng","hitam putih","Afrika","sabana","kuda","herbivora","kawanan",
              "singa","predator","berlari","cepat","kaki kuat","migras","serengeti",
              "masai mara","kenya","tanzania","africa","zebra gunung","zebra dataran",
              "grevy zebra","zebra quagga","punah","dilindungi","simbol","unik"],

    "jerapah": ["tinggi","leher panjang","Afrika","daun","akasia","herbivora",
                "berbintik","coklat kuning","berlari","cepat","kaki panjang",
                "jantung besar","tekanan darah","hewan tertinggi","taman nasional",
                "savana","padang","pohon akasia","makan daun","lidah panjang","biru"],

    "lumba": ["ikan paus","laut","samudra","cerdas","jinak","melompat","bermain",
              "lumba-lumba","lumba irrawaddy","lumba hidung botol","lumba spinner",
              "mamalia laut","cetacean","bernapas udara","sirip","echolocation",
              "sonar","berkomunikasi","cerdas","sosial","kawanan","pod","sekolah",
              "akrobat","show","pelatihan","aquarium","sea world","konservasi"],

    "paus": ["besar","laut","samudra","mamalia","cetacean","paus biru","paus sperma",
             "paus humpback","paus fin","paus minke","paus orca","paus beluga",
             "paus narwhal","paus balin","terbesar","hewan terbesar","bernapas",
             "lubang udara","blowhole","migrasi","lagu paus","berkomunikasi",
             "konservasi","perburuan paus","moratorium","IWC","punah"],

    "hiu": ["predator","berbahaya","laut","samudra","gigi","rahang","sirip","cepat",
            "hiu putih","hiu harimau","hiu banteng","hiu paus","hiu martil",
            "hiu biru","hiu lemon","hiu karang","hiu blacktip","hiu whitetip",
            "serangan hiu","berbahaya","darah","indera penciuman","electroreception",
            "lateral line","top predator","ekosistem laut","konservasi"],

    "elang": ["terbang","tinggi","paruh","cakar","tajam","memburu","mangsa","bebas",
              "elang jawa","elang bondol","elang hitam","elang brontok","elang ular",
              "elang bido","elang laut","elang botak","bald eagle","rajawali",
              "simbol","nasional","Amerika","Indonesia","pemandangan","indah",
              "sayap lebar","terbang tinggi","melayang","soaring","gliding"],

    "lebah": ["madu","sarang","sengat","bunga","nektar","royal jelly","pollen",
              "serbuk sari","ratu lebah","pekerja","drone","koloni","kawanan",
              "lebah madu","lebah bumblebee","lebah soliter","lebah tanah",
              "apis mellifera","apis cerana","penyerbukan","polinator","ekosistem",
              "pertanian","produksi pangan","propolis","wax","lilin lebah"],

    "kupu": ["cantik","sayap","warna-warni","terbang","bunga","metamorfosis","ulat",
             "kepompong","chrysalis","pupa","larva","telur","siklus hidup","nektar",
             "penyerbukan","taman","kebun","alam","indah","ringan","kupu-kupu",
             "kupu raja","kupu swallowtail","kupu morpho","kupu monarch","migration"],

    "semut": ["kecil","kuat","koloni","ratu","pekerja","prajurit","sarang","gula",
              "makanan","baris","teratur","sosial","kerja sama","komunal","serangga",
              "semut api","semut hitam","semut merah","semut putih","rayap","laron",
              "sarang semut","antisocial","feromin","komunikasi kimia","organizer"],

    "lele": ["ikan air tawar","sungai","kolam","tambak","budidaya","ternak ikan",
             "makanan","lele goreng","lele bakar","lele asap","protein","murah",
             "lele dumbo","lele lokal","lele sangkuriang","berkumis","kumis",
             "catfish","omnivora","pemakan segala","tahan banting","mudah dibudidaya"],

    "udang": ["laut","sungai","tambak","budidaya","seafood","makanan","protein",
              "udang windu","udang vaname","udang galah","udang putih","udang api",
              "kupas udang","udang goreng","udang bakar","capit","cangkang","kulit",
              "merah saat dimasak","fresh","segar","beku","ekspor"],

    "kepiting": ["capit","cangkang","laut","pantai","bakau","mangrove","seafood",
                 "kepiting bakau","kepiting rajungan","kepiting kenari","kepiting biru",
                 "rajungan","capit kuat","berjalan menyamping","bersembunyi","pasir",
                 "lumpur","air payau","kepiting goreng","kepiting asam manis"],

    "penyu": ["laut","telur","pantai","bertelur","dilindungi","langka","tua","panjang umur",
              "penyu hijau","penyu sisik","penyu lekang","penyu belimbing","penyu pipih",
              "cangkang","karapas","bermigrasi","menavigasi","arus laut","kura laut",
              "tukik","anak penyu","menetas","pasir pantai","konservasi","WWF"],

    "kumbang": ["serangga","cangkang","keras","elytra","antena","terbang","kayu",
                "kumbang tanduk","kumbang stag","kumbang gajah","kumbang madu",
                "ladybug","kumbang merah","bintik hitam","kumbang kayu","hama",
                "kumbang beras","kumbang gudang","larva","ulat","pupa","metamorfosis"],

    "capung": ["sayap","terbang","cepat","air","kolam","sawah","capung biru",
               "capung merah","capung hijau","dragonfly","damselfly","nimfa",
               "larva air","metamorfosis","memangsa nyamuk","predator nyamuk",
               "ekosistem sawah","indikator kualitas air","serangga kuno"],

    # ══════════════════════════════════════════════════════════
    # TUMBUHAN
    # ══════════════════════════════════════════════════════════
    "mawar": ["bunga","merah","mekar","harum","cinta","taman","duri","kelopak",
              "berduri","romantis","hadiah","buket","mawar merah","mawar putih",
              "mawar kuning","mawar pink","taman mawar","parfum mawar","minyak mawar",
              "rose water","produk kecantikan","simbol cinta","valentine","ulang tahun"],

    "melati": ["bunga","putih","harum","Indonesia","nasional","pengantin","bunga melati",
               "merangkai","karangan bunga","taman","kebun","wangi","aroma","minyak melati",
               "jasmine","bunga nasional","pernikahan adat","upacara","tradisi"],

    "anggrek": ["bunga","indah","eksotis","langka","hias","anggrek bulan","anggrek hitam",
                "phalaenopsis","dendrobium","cattleya","vanda","oncidium","epidendrum",
                "koleksi","hobi","pembudidayaan","greenhouse","pot","media tanam",
                "pupuk","penyiraman","sinar matahari","kelembaban","suhu"],

    "bambu": ["rotan","kayu","hijau","kuat","lentur","cepat tumbuh","serangga",
              "bambu betung","bambu apus","bambu kuning","bambu hitam","bambu tali",
              "kerajinan bambu","mebel bambu","alat musik","angklung","calung",
              "kulintang","bambu sebagai bahan bangunan","rumah bambu","gazebo",
              "tumbuhan cepat tumbuh","ramah lingkungan","panda makan bambu"],

    "jati": ["kayu","keras","kuat","mahal","mebel","furniture","kayu jati",
             "pohon jati","jati perhutani","jati rakyat","kayu mahal","kayu berkualitas",
             "kayu ekspor","kursi jati","meja jati","lemari jati","dipan jati",
             "ketahanan","awet","anti rayap","tahan air","tahan cuaca","warisan"],

    "pinus": ["cemara","pohon","hijau","tinggi","hutan pinus","gunung","pegunungan",
              "dingin","sejuk","segar","aroma pinus","hutan hijau","biji pinus",
              "pine cone","buah pohon pinus","getah pinus","turpentine","terpentin",
              "kayu pinus","konstruksi","kertas","pulp","celulosa","industri"],

    "cemara": ["pinus","pohon","hijau","Natal","dekorasi","pohon cemara","cemara wangi",
               "cemara udang","cemara gunung","cemara laut","pantai","angin","tahan"],

    "beringin": ["pohon besar","tua","rindang","akar udara","akar gantung","keramat",
                 "sakral","tempat berkumpul","desa","alun-alun","warung","cerita rakyat",
                 "hantu","mitos","legenda","banyan tree","ficus","pohon suci","perlindungan"],

    "mangga": ["buah","manis","asam","kuning","merah","hijau","musim","panen",
               "mangga harum manis","mangga gedong","mangga arumanis","mangga manalagi",
               "mangga golek","mangga indramayu","mangga alpukat","mangga kelapa",
               "rujak","jus mangga","manisan mangga","asinan mangga","pickle mango"],

    "rambutan": ["buah","merah","berduri","manis","lokal","tropis","lebat","panen",
                 "rambutan binjai","rambutan rapiah","rambutan lebak bulus","si manis",
                 "buah tropis","biji rambutan","kulit rambutan","dimakan segar",
                 "es buah","jus buah","buah kaleng","ekspor buah"],

    "durian": ["buah","bau","duri","manis","raja buah","musang king","monthong",
               "durian lokal","durian Bangkok","aroma kuat","tidak disukai semua",
               "Durian Medan","durian Pontianak","durian Palembang","festival durian",
               "es durian","pancake durian","kue durian","lempok durian","dodol"],

    "pisang": ["buah","kuning","manis","goreng","pisang goreng","pisang rebus",
               "pisang kepok","pisang raja","pisang ambon","pisang tanduk","pisang cavendish",
               "getah pisang","batang pisang","daun pisang","bunga pisang","jantung pisang",
               "kolak pisang","kripik pisang","sale pisang","brownis pisang"],

    "pepaya": ["buah","manis","jus","pepaya matang","pepaya muda","sayur pepaya",
               "tumis pepaya","acar pepaya","biji pepaya","daun pepaya","getah pepaya",
               "papain","enzim","pencernaan","vitamin C","beta karoten","kesehatan"],

    "kelapa": ["santan","minyak","air kelapa","sabut","batok","lidi","daun kelapa",
               "pohon kelapa","kelapa muda","kelapa tua","kelapa parut","kelapa bakar",
               "es kelapa","dawet","cendol","opor","rendang","gulai","masak santan",
               "industri kelapa","kopra","minyak kelapa","virgin coconut oil"],

    "padi": ["sawah","beras","nasi","petani","panen","masa tanam","irigasi","air",
             "padi IR64","padi ciherang","padi rojolele","padi basmati","padi japonica",
             "gabah","dedak","bekatul","klobot","jerami","pupuk","pestisida",
             "hama wereng","hama tikus","hama burung","musim tanam","masa panen"],

    "jagung": ["padi","singkong","ubi","beras","makanan pokok","petani","kebun",
               "jagung manis","jagung pipil","pop corn","jagung bakar","jagung rebus",
               "corn","biji jagung","tongkol","rambut jagung","kulit jagung","pati jagung",
               "tepung jagung","maizena","sirup jagung","industri pangan"],

    "singkong": ["ubi","ketela","pohon","kayu","umbi","pati","tepung","tapioka",
                 "getuk","tiwul","gaplek","tape singkong","singkong goreng","singkong rebus",
                 "kerupuk singkong","keripik singkong","brownies singkong","bolu singkong"],

    "kentang": ["ubi","umbi","tanah","perlu","kentang goreng","kentang rebus","kentang tumbuk",
                "mashed potato","french fries","potato chips","pati","tepung kentang",
                "kentang baby","kentang granola","sop kentang","rendang kentang","curry"],

    "wortel": ["sayuran","oranye","vitamin A","mata","sehat","rebus","goreng",
               "wortel parut","jus wortel","wortel rebus","sup wortel","soto wortel",
               "acar wortel","carrot","wortel mini","wortel besar","beta karoten"],

    "bayam": ["sayuran","hijau","spinach","vitamin","zat besi","sehat","masak",
              "bayam merah","bayam hijau","sayur bayam","bobor bayam","bening bayam",
              "gado-gado","pecel","lalapan","bayam organik","bayam wild","berdaun"],

    "cabai": ["pedas","merah","rawit","saus","sambal","bumbu","masak","panas",
              "cabai rawit","cabai merah","cabai hijau","cabai besar","paprika",
              "capsaicin","pedas sekali","level kepedasan","scoville","jengkol cabai",
              "cabe tumbuk","cabe giling","sambal terasi","sambal bajak"],

    "tomat": ["merah","bulat","sayuran","saus tomat","jus tomat","tomat ceri",
              "lycopene","vitamin C","antioksidan","sayur tomat","sup tomat",
              "tomat goreng","masakan","bumbu","spaghetti","pizza","ketchup"],

    "bawang": ["bawang merah","bawang putih","bawang bombay","masak","bumbu","pedas",
               "menangis","iris bawang","goreng bawang","bawang goreng","topping",
               "daun bawang","daun kucai","allicin","kesehatan","anti inflamasi"],

    "jahe": ["rempah","panas","pedas","jahe merah","jahe emprit","jahe gajah",
             "minuman jahe","wedang jahe","bandrek","bajigur","susu jahe","kopi jahe",
             "ginger","anti mual","kesehatan","imunitas","anti inflamasi","masak"],

    "kunyit": ["rempah","kuning","jahe","warna","masak","bumbu","kunyit asam",
               "kunir","jamu","kesehatan","anti kanker","curcumin","curcuminoid",
               "pewarna alami","masakan kuning","rendang","opor","gulai","curry"],

    "kopi": ["minum","kafein","hitam","pahit","panas","perkebunan","roasting",
             "kopi arabika","kopi robusta","kopi liberika","kopi excelsa","kopi luwak",
             "espresso","cappuccino","latte","americano","cortado","cold brew",
             "kopi tubruk","kopi susu","kopi sachet","nescafe","kopi instan"],

    "teh": ["minum","panas","hijau","hitam","herbal","teh tubruk","teh celup",
            "teh tarik","teh manis","teh tawar","teh es","teh melati","teh chamomile",
            "green tea","black tea","white tea","oolong tea","pu-erh","matcha",
            "teh manis","teh pahit","teh lemon","teh jahe","teh herbal"],

    "salak": ["buah","coklat","bersisik","manis","asam","salak pondoh","salak bali",
              "buah tropis","lokal","khas","bijinya besar","daging buah","buah segar",
              "manisan salak","keripik salak","dodol salak","minuman salak"],

    "manggis": ["buah","ungu","putih","manis","ratu buah","kulit manggis","xanthone",
                "antioksidan","kesehatan","manggis segar","jus manggis","suplemen",
                "obat herbal","khasiat manggis","buah tropis","lokal","panen"],

    "alpukat": ["buah","hijau","lembut","lemak baik","vitamin","alpukat mentega",
                "alpukat hass","alpukat fuerte","jus alpukat","es alpukat","guacamole",
                "toast alpukat","salad","smoothie","omega 9","asam lemak tak jenuh"],

    "nanas": ["buah","kuning","asam","manis","tropical","pineapple","bromelain",
              "nanas madu","nanas bogor","nanas subang","jus nanas","sirup nanas",
              "selai nanas","dodol nanas","kue nanas","campuran masakan","sayur"],

    "semangka": ["buah","merah","hijau","segar","air","manis","musim panas","biji",
                 "semangka tanpa biji","semangka kuning","jus semangka","es semangka",
                 "lycopene","antioksidan","menyegarkan","buah musim panas","pantai"],

    "jamur": ["fungi","tumbuh","lembab","hutan","kayu","tanah","jamur tiram","jamur kuping",
              "jamur merang","jamur kancing","mushroom","shiitake","portobello","truffle",
              "jamur liar","beracun","tidak beracun","edible","payung","spora","miselia"],

    # ══════════════════════════════════════════════════════════
    # MAKANAN & MINUMAN
    # ══════════════════════════════════════════════════════════
    "nasi": ["beras","makan","lauk","pauk","nasi putih","nasi goreng","nasi uduk",
             "nasi kuning","nasi liwet","nasi padang","nasi kebuli","nasi timbel",
             "nasi bakar","nasi bogana","lontong","ketupat","bubur","nasi lemak",
             "nasi biryani","staple food","makanan pokok","indonesia","asia"],

    "sate": ["daging","bumbu","tusuk","bakar","sate ayam","sate kambing","sate sapi",
             "sate lilit","sate padang","sate madura","sate babi","lontong sate",
             "bumbu kacang","saus kecap","sambal bawang","kecap manis","arang","bakar"],

    "bakso": ["daging","kuah","mie","bihun","tofu","tahu","bakso sapi","bakso ikan",
              "bakso ayam","bakso goreng","bakso bakar","siomay","batagor","cilok",
              "cimol","cireng","tekwan","pempek","kuah bening","kuah merah","pedas"],

    "soto": ["kuah","daging","ayam","sapi","soto ayam","soto betawi","soto lamongan",
             "soto mie","coto makassar","soto kudus","soto banjar","soto medan",
             "tauge","soun","telur","kentang","emping","kerupuk","koya","sambal"],

    "rendang": ["padang","daging","pedas","sapi","masak lama","bumbu rempah","masak kering",
                "rendang basah","rendang kering","rendang ayam","rendang paru","rendang rebung",
                "santan","cabai","kunyit","jahe","serai","daun jeruk","daun kunyit"],

    "tempe": ["kedelai","fermentasi","protein","murah","bergizi","tempe goreng","tempe bacem",
              "tempe mendoan","tempe kering","tempe orek","tempe penyet","kedelai",
              "rhizopus","ragi tempe","produk lokal","makanan tradisional","vegetarian"],

    "tahu": ["kedelai","putih","lunak","tahu goreng","tahu bacem","tahu isi","tahu sumedang",
             "tahu bandung","tahu pong","tahwa","tofu","protein nabati","vegetarian",
             "isoflavon","masak","tumis","sup","bacem","sayur","pecel","gado"],

    "mie": ["kuah","goreng","mie ayam","mie goreng","mie rebus","mie pangsit","indomie",
            "bakmi","kwetiau","bihun","soun","mie tek-tek","mie aceh","mie celor",
            "ramen","udon","soba","pasta","spaghetti","tepung terigu","telur"],

    "roti": ["gandum","tepung","bakar","panggang","roti tawar","roti sobek","roti manis",
             "roti coklat","roti keju","croissant","baguette","sourdough","whole wheat",
             "roti lapis","sandwich","toast","mentega","selai","kaya","nutella"],

    "kue": ["manis","tepung","gula","telur","mentega","kue tar","kue basah","kue kering",
            "brownies","muffin","cupcake","donat","kue lapis","klepon","onde-onde",
            "risoles","lemper","nagasari","putri salju","nastar","kastengel"],

    "susu": ["minum","putih","segar","bernutrisi","susu sapi","susu kambing","susu kedelai",
             "susu almond","susu oat","susu UHT","susu pasteurisasi","ASI","formula",
             "keju","mentega","yogurt","es krim","krim","whey","kasein","laktosa"],

    "gula": ["manis","tebu","pasir","putih","merah","aren","semut","gula pasir",
             "gula merah","gula jawa","gula aren","gula batu","gula halus","sukrosa",
             "fruktosa","glukosa","diabetes","kalori","pemanis","sirup","madu"],

    "garam": ["asin","mineral","laut","garam dapur","garam laut","garam himalaya",
              "natrium","klorida","bumbu","masak","pengawet","elektrolit","tubuh",
              "air laut","penguapan","petambak","produksi garam","garam yodium"],

    "madu": ["lebah","manis","emas","sarang","nektar","royal jelly","propolis","pollen",
             "madu hutan","madu trigona","madu randu","madu kapuk","madu sialang",
             "kesehatan","antibakteri","antioksidan","pemanis alami","obat tradisional"],

    "sambal": ["pedas","cabai","tomat","bawang","terasi","sambal terasi","sambal bawang",
               "sambal ijo","sambal matah","sambal bajak","sambal goreng","sambal merah",
               "sambal kacang","sambal mangga","ulek","cobek","level pedas","panas"],

    "kecap": ["manis","asin","hitam","kedelai","kecap manis","kecap asin","kecap ikan",
              "kecap bango","kecap ABC","masak","bumbu","sate","rendang","semur",
              "teriyaki","barbecue","marinasi","celup","topping","saus"],

    "telur": ["ayam","rebus","goreng","dadar","ceplok","mata sapi","orak arik","scrambled",
              "telur asin","telur pindang","telur balado","protein","kuning telur","putih telur",
              "omelet","quiche","souffle","meringue","telur bebek","telur puyuh"],

    "daging": ["sapi","ayam","kambing","babi","daging merah","daging putih","protein",
               "lemak","rendang","sate","sop","gulai","semur","steak","panggang",
               "rebus","goreng","tumis","daging cincang","daging giling","bakso"],

    # ══════════════════════════════════════════════════════════
    # PROFESI
    # ══════════════════════════════════════════════════════════
    "dokter": ["kesehatan","rumah sakit","pasien","obat","diagnosis","operasi",
               "dokter umum","dokter spesialis","dokter gigi","dokter anak","bedah",
               "penyakit","memeriksa","resep","klinik","puskesmas","apotek",
               "stetoskop","mantel putih","alat medis","USG","rontgen","lab"],

    "guru": ["mengajar","murid","siswa","kelas","sekolah","pelajaran","kurikulum",
             "guru SD","guru SMP","guru SMA","dosen","pengajar","pendidik","tutor",
             "papan tulis","buku","pena","spidol","nilai","ujian","PR","tugas",
             "wisuda","ijazah","ilmu","pengetahuan","mendidik","membimbing"],

    "polisi": ["hukum","keamanan","menjaga","menangkap","penjahat","seragam","biru",
               "polisi lalu lintas","polisi kriminal","polresta","polda","Polri",
               "brimob","intel","reserse","SIM","STNK","tilang","razia","patroli",
               "senjata","borgol","mobil patroli","pos polisi","kantor polisi"],

    "tentara": ["militer","perang","senjata","seragam","loreng","baret","pangkat",
                "TNI","TNI AD","TNI AL","TNI AU","jenderal","kolonel","mayor",
                "kapten","letnan","sersan","kopral","prajurit","bela negara",
                "pangkalan","markas","barak","latihan","kedisiplinan","patriot"],

    "petani": ["sawah","ladang","bercocok tanam","panen","padi","jagung","sayuran",
               "cangkul","bajak","traktor","pupuk","pestisida","irigasi","saluran air",
               "musim tanam","musim panen","harga padi","tengkulak","koperasi tani",
               "pertanian","agrikultur","desa","pedesaan","tanah","kesuburan"],

    "nelayan": ["ikan","laut","perahu","jaring","pancing","joran","kail","umpan",
                "pagi buta","berangkat fajar","pulang sore","ikan segar","hasil laut",
                "pelabuhan","TPI","tempat pelelangan ikan","nelayan tradisional",
                "kapal motor","kapal ikan","cuaca","badai","ombak","berbahaya"],

    "pedagang": ["jual","beli","pasar","toko","warung","jualan","dagang","barang",
                 "untung","rugi","modal","omzet","pedagang kecil","pedagang besar",
                 "wholesale","retail","grosir","eceran","toko kelontong","PKL",
                 "kaki lima","online shop","marketplace","shopee","tokopedia"],

    "pengacara": ["hukum","klien","sidang","pengadilan","hakim","jaksa","terdakwa",
                  "membela","kasus","hukum pidana","hukum perdata","kontrak","notaris",
                  "LBH","firma hukum","law firm","advokat","konsultan hukum"],

    "hakim": ["pengadilan","vonis","putusan","hukum","sidang","jaksa","pengacara",
              "terdakwa","saksi","bukti","pasal","KUHP","mahkamah agung","MA",
              "PT","PN","mahkamah konstitusi","KPK","keadilan","objektif"],

    "arsitek": ["desain","bangunan","gedung","denah","blueprint","konstruksi","struktur",
                "estetika","fungsional","arsitektur","autocad","revit","sketsa","model",
                "bangunan modern","bangunan tradisional","interior","eksterior","urban"],

    "pilot": ["pesawat","terbang","kokpit","landasan","menerbangkan","navigasi",
              "cuaca","ketinggian","kecepatan","kargo","penumpang","maskapai",
              "seragam","lisensi","SIM terbang","simulator","ATL","PPL","CPL"],

    "sopir": ["mobil","mengemudi","kemudi","rem","gas","jalan","penumpang","supir",
              "sopir taksi","sopir ojol","sopir bus","sopir angkot","sopir pribadi",
              "SIM","STNK","bensin","parkir","macet","jalur","tol","lampu merah"],

    "koki": ["masak","dapur","resep","masakan","chef","kuliner","rasa","bumbu",
             "koki profesional","chef executive","sous chef","pastry chef","saucier",
             "restoran","hotel","catering","masak enak","bumbu rahasia","teknik masak"],

    "penulis": ["menulis","buku","novel","cerpen","puisi","karya","sastra","kata",
                "keyboard","laptop","naskah","cerita","fantasi","fiksi","non-fiksi",
                "jurnalis","wartawan","editor","penerbit","ISBN","royalti","bestseller"],

    "seniman": ["seni","melukis","menggambar","patung","instalasi","karya seni",
                "galeri","pameran","kanvas","cat","kuas","studio","kreatif","ekspresi",
                "seniman modern","seniman tradisional","pelukis","pematung","desainer"],

    "musisi": ["musik","bermain","alat musik","gitar","piano","drum","biola","vokal",
               "konser","album","recording","studio musik","band","orkestra","jazz",
               "pop","rock","dangdut","keroncong","indie","kolaborasi","soundcheck"],

    "atlet": ["olahraga","latihan","kompetisi","juara","medali","trophy","prestasi",
              "atlet profesional","atlet amatir","pelatih","fisik","mental","diet",
              "sparing","turnamen","olimpiade","sea games","asian games","PON"],

    # ══════════════════════════════════════════════════════════
    # TUBUH MANUSIA
    # ══════════════════════════════════════════════════════════
    "kepala": ["rambut","otak","wajah","telinga","mata","hidung","mulut","leher",
               "tengkorak","tulang kepala","kepala sakit","pusing","migrain","kepala pening",
               "helm","topi","sakit kepala","berpikir","akal","pikiran","kecerdasan"],

    "mata": ["melihat","penglihatan","kacamata","lensa","retina","kornea","iris","pupil",
             "mata minus","mata plus","rabun","buta","buta warna","katarak","glaukoma",
             "air mata","menangis","berkedip","menatap","memandang","indra penglihatan"],

    "hidung": ["mencium","bau","penciuman","lubang hidung","tulang hidung","cuping",
               "pilek","flu","tersumbat","mendengus","mengendus","parfum","aroma",
               "wangi","busuk","sedap","tidak sedap","hidung mancung","hidung pesek"],

    "mulut": ["bicara","makan","gigi","lidah","bibir","rahang","tenggorokan","ludah",
              "berbicara","menyanyi","meniup","mencium","berteriak","berbisik","oral",
              "gusi","rongga mulut","kesehatan gigi","odol","sikat gigi","obat kumur"],

    "telinga": ["mendengar","suara","pendengaran","gendang telinga","telinga kanan",
                "telinga kiri","tuli","kurang dengar","earphone","headphone","tinnitus",
                "kotoran telinga","daun telinga","saluran telinga","cuping telinga"],

    "tangan": ["jari","telapak","jempol","genggam","pegang","dorong","tarik","jari telunjuk",
               "jari tengah","jari manis","jari kelingking","kuku","gelang","cincin",
               "jam tangan","pergelangan","tangan kanan","tangan kiri","jabat tangan"],

    "kaki": ["berjalan","berlari","telapak kaki","tumit","jari kaki","lutut","paha",
             "betis","tumit","sepatu","sandal","kaos kaki","kuku kaki","pijat kaki",
             "kaki kanan","kaki kiri","kaki gajah","kaki lemas","kelelahan"],

    "jantung": ["detak","pompa","darah","berdetak","sirkulasi","kardiovaskuler",
                "jantung koroner","serangan jantung","penyakit jantung","EKG","ECG",
                "operasi jantung","bypass","katup jantung","jantung sehat","olahraga",
                "kolesterol","tekanan darah","hipertensi","hipotensi","sehat"],

    "paru": ["bernapas","oksigen","udara","paru-paru","paru kanan","paru kiri",
             "bronkus","alveolus","kapasitas paru","TBC","pneumonia","asma","COPD",
             "kanker paru","merokok","bahaya rokok","udara bersih","polusi udara"],

    "otak": ["berpikir","cerdas","saraf","neuron","memori","ingatan","pikiran","akal",
             "otak kiri","otak kanan","lobus","korteks","cerebrum","cerebellum",
             "batang otak","kecerdasan","IQ","EQ","belajar","memproses informasi"],

    "tulang": ["rangka","keras","kalsium","mineral","sendi","tulang belakang","tulang iga",
               "tulang rusuk","tengkorak","tulang paha","tulang lengan","tulang kering",
               "tulang rawan","kartilago","osteoporosis","patah tulang","pertumbuhan"],

    "kulit": ["luar","perlindungan","sel","dermis","epidermis","kulit putih","kulit hitam",
              "coklat","kulit sehat","kulit bermasalah","jerawat","bekas luka","kulit kering",
              "kulit berminyak","moisturizer","sunscreen","lotion","krim","sabun"],

    "darah": ["merah","mengalir","vena","arteri","jantung","hemoglobin","sel darah",
              "golongan darah","A","B","AB","O","transfusi","donor darah","PMI",
              "luka","perdarahan","tekanan darah","anemia","leukemia","platelet"],

    "napas": ["bernapas","udara","oksigen","paru","menghirup","menghembuskan","hidung",
              "mulut","dada","pernapasan","respirasi","trakea","bronkus","alveolus",
              "nafas panjang","nafas dalam","sesak napas","asma","meditasi","yoga"],

    "rambut": ["kepala","hitam","panjang","pendek","keriting","lurus","potong","salon",
               "sampo","kondisioner","rambut rontok","kebotakan","semir","warna rambut",
               "rambut pirang","rambut coklat","rambut merah","rambut putih","uban"],

    "perut": ["makan","lapar","kenyang","pencernaan","lambung","usus","perut kembung",
              "diare","sakit perut","perut mulas","sembelit","mual","muntah","gas",
              "asam lambung","maag","GERD","enzim pencernaan","flora usus","probiotik"],

    "dada": ["jantung","paru","tulang rusuk","sternum","payudara","dada lebar","dada bidang",
             "sakit dada","sesak dada","nyeri dada","serangan jantung","paru","napas"],

    # ══════════════════════════════════════════════════════════
    # PERASAAN & EMOSI
    # ══════════════════════════════════════════════════════════
    "cinta": ["kasih","sayang","rindu","menikah","pasangan","kekasih","hati","romantis",
              "cinta tulus","cinta sejati","jatuh cinta","patah hati","putus","selingkuh",
              "pernikahan","hubungan","pacaran","comblang","kencan","valentines",
              "bunga","hadiah","coklat","ciuman","pelukan","romantis","mesra"],

    "rindu": ["kangen","jauh","berpisah","ingin bertemu","savoring","nostalgia",
              "kenangan","masa lalu","masa kecil","teman lama","surat","telepon",
              "merajuk","menunggu","kesepian","sendirian","rindu berat","rindu teramat"],

    "sedih": ["menangis","duka","berduka","depresi","galau","susah","menderita",
              "kehilangan","ditinggalkan","patah hati","kecewa","frustrasi","putus asa",
              "air mata","sesak","berat hati","galau","mengurung diri","murung"],

    "bahagia": ["senang","gembira","riang","ceria","suka cita","puas","syukur",
                "tertawa","senyum","bersemangat","bersyukur","beruntung","berhasil",
                "sukses","merayakan","pesta","bahagia sejati","ketenangan","damai hati"],

    "marah": ["emosi","naik darah","marah-marah","teriak","membentak","melampiaskan",
              "frustrasi","jengkel","kesal","dongkol","geram","naik pitam","meledak",
              "mengeluarkan kata kasar","amarah","kebencian","dendam","sakit hati"],

    "takut": ["fobia","ngeri","gemetar","panik","cemas","khawatir","teror","horor",
              "gelap","sendirian","bayangan","mimpi buruk","phobia","arachnofobia",
              "klaustrofobia","acrofobia","social anxiety","gugup","grogi"],

    "bangga": ["prestasi","pencapaian","sukses","berhasil","menang","juara","meraih",
               "kebanggaan","bangga diri","bangga negara","nasionalisme","patriotisme",
               "apresiasi","pengakuan","penghargaan","trofi","medali","sertifikat"],

    "malu": ["merah","pipi","malu-malu","sungkan","canggung","awkward","tidak percaya diri",
             "rendah diri","insecure","takut penilaian","social anxiety","introvert",
             "pemalu","pendiam","menyendiri","tidak suka pusat perhatian"],

    "senang": ["bahagia","gembira","riang","ceria","suka","tertawa","senyum","enjoy",
               "menikmati","puas","memuaskan","menggembirakan","menyenangkan","fun",
               "happy","joyful","ecstatic","euphoria","bersemangat","antusias"],

    # ══════════════════════════════════════════════════════════
    # TEMPAT & BANGUNAN
    # ══════════════════════════════════════════════════════════
    "rumah": ["tinggal","keluarga","tempat berteduh","atap","dinding","pintu","jendela",
              "kamar","dapur","ruang tamu","kamar mandi","garasi","taman","pagar",
              "rumah minimalis","rumah type 36","rumah type 45","perumahan","KPR",
              "cicilan","sertifikat","IMB","kontrakan","kos","apartemen","mes"],

    "sekolah": ["belajar","murid","guru","kelas","pelajaran","ujian","tugas","buku","rapor","nilai",
                "SD","SMP","SMA","universitas","kampus","pendidikan","kurikulum",
                "kantin","perpustakaan","lapangan","aula","UKS","seragam","upacara"],

    "masjid": ["sholat","ibadah","Islam","adzan","imam","jamaah","mimbar","mihrab",
               "kubah","menara","wudhu","thaharah","Jumat","sholat Jumat","tarawih",
               "ramadan","idul fitri","idul adha","quran","ceramah","takmir"],

    "gereja": ["ibadah","Kristen","Katolik","pendeta","pastor","kebaktian","misa",
               "baptis","natal","paskah","salib","menara","lonceng","doa","nyanyian",
               "jemaat","alkitab","injil","sakramen","baptis","sidi","pernikahan"],

    "istana": ["raja","ratu","kerajaan","megah","mewah","besar","indah","sejarah",
               "istana negara","istana presiden","istana bogor","istana cipanas",
               "keraton","kasultanan","sultan","kebudayaan","wisata sejarah","heritage"],

    "pasar": ["jual","beli","pedagang","penjual","pembeli","tawar","harga","murah",
              "pasar tradisional","pasar modern","supermarket","mall","warung","toko",
              "sayur","buah","daging","ikan","bumbu","rempah","barang","ramai"],

    "kantor": ["bekerja","pegawai","karyawan","meja","kursi","komputer","printer",
               "rapat","meeting","atasan","bawahan","jam kerja","absen","lembur",
               "kantor pemerintah","kantor swasta","BUMN","startup","gedung perkantoran"],

    "hotel": ["menginap","kamar","malam","check in","check out","resepsionis","lobby",
              "restoran","kolam renang","spa","sauna","gym","laundry","room service",
              "hotel bintang","budget hotel","hostel","resort","villa","penginapan"],

    "perpustakaan": ["buku","membaca","koleksi","rak buku","membaca","tenang","hening",
                     "referensi","perpustakaan nasional","perpustakaan daerah","perpus kampus",
                     "katalog","ISBN","DDC","pustakawan","digital library","e-book"],

    "stadion": ["olahraga","sepak bola","penonton","tribun","lapangan","gol","pertandingan",
                "kompetisi","turnamen","liga","cup","trophy","atlet","pemain","wasit",
                "GBK","stadion utama","stadion bung karno","gelora bung karno"],

    "bandara": ["pesawat","terbang","penumpang","tiket","check in","boarding","terminal",
                "landasan","runway","taxiway","apron","ATC","air traffic control",
                "imigrasi","bea cukai","kargo","bagasi","maskapai","penerbangan"],

    "pelabuhan": ["kapal","laut","bongkar muat","kontainer","feri","penyeberangan",
                  "nelayan","terminal penumpang","terminal kargo","ekspor","impor",
                  "bea cukai","imigrasi","karantina","dermaga","jetty","tanjung priok"],

    "jembatan": ["sungai","laut","menghubungkan","konstruksi","baja","beton","cable stayed",
                 "suspensi","jembatan gantung","fly over","overpass","underpass","viaduct",
                 "jembatan terpanjang","nasuru","suramadu","ampera","kutai kartanegara"],

    "menara": ["tinggi","Eiffel","TVRI","menara TV","menara listrik","menara BTS",
               "menara masjid","minaret","bangunan tinggi","pencakar langit","skyscraper",
               "WTC","burj khalifa","CN tower","menara kembar","Petronas tower"],

    "taman": ["hijau","bunga","pohon","rumput","duduk","santai","jalan-jalan","piknik",
              "taman kota","taman nasional","kebun raya","taman bermain","playground",
              "bangku taman","air mancur","kolam","lampu taman","gazebo","pergola"],

    # ══════════════════════════════════════════════════════════
    # TRANSPORTASI
    # ══════════════════════════════════════════════════════════
    "mobil": ["kendaraan","beroda empat","bensin","sopir","kemudi","rem","gas","klakson",
              "sedan","SUV","MPV","pickup","truk","van","jeep","sport","mewah",
              "Toyota","Honda","Suzuki","Daihatsu","Mitsubishi","Ford","BMW","Mercedes"],

    "motor": ["sepeda motor","beroda dua","bensin","berkendara","helm","seragam",
              "motor matic","motor manual","motor sport","motor bebek","motor besar",
              "Honda","Yamaha","Suzuki","Kawasaki","KTM","Ducati","Harley","BSA"],

    "kapal": ["laut","berlayar","lautan","samudra","pelabuhan","nakhoda","awak kapal",
              "kapal penumpang","kapal kargo","kapal tanker","kapal perang","kapal selam",
              "feri","yacht","kapal pesiar","cruise ship","kapal nelayan","perahu"],

    "pesawat": ["terbang","bandara","pilot","penumpang","tiket","cabin","kokpit","mesin",
                "sayap","turbine","jet","baling-baling","propeller","boeing","airbus",
                "Garuda","Lion Air","Citilink","Batik Air","Sriwijaya","Super Air"],

    "kereta": ["rel","stasiun","gerbong","masinis","penumpang","tiket","KRL","MRT","LRT",
               "kereta cepat","kereta ekspres","kereta api","DAOP","KAI","commuter line",
               "Argo Bromo","Argo Jati","Taksaka","kereta tua","lokomotif","uap"],

    "sepeda": ["kayuh","roda","setang","rem","pelek","ban","sadel","pedal","gear",
               "sepeda gunung","sepeda balap","sepeda santai","sepeda listrik","BMX",
               "gowes","bersepeda","jalur sepeda","helmnya","baju sepeda","komunitas"],

    "perahu": ["air","dayung","layar","nelayan","sungai","danau","laut","kano","kayak",
               "sampan","gondola","rakit","jukung","motor tempel","speedboat","kapal kecil",
               "berlayar","mendayung","arus","ombak","angin","layar","tiang"],

    "bus": ["angkutan","penumpang","jalan","sopir","trayek","terminal","halte","ongkos",
            "bus kota","bus antar kota","bus AKAP","bus malam","bus pariwisata",
            "Trans Jakarta","Trans Metro","Damri","PO Rosalia","PO Haryanto"],

    "taksi": ["penumpang","argometer","sopir","AC","nyaman","kota","taksi online",
              "Gojek","Grab","Maxim","inDriver","Blue Bird","Express","Silver Bird",
              "argo","tarif","meter","booking","ojek","ojol","ride sharing"],

    # ══════════════════════════════════════════════════════════
    # TEKNOLOGI
    # ══════════════════════════════════════════════════════════
    "komputer": ["laptop","PC","desktop","keyboard","mouse","monitor","CPU","RAM",
                 "hardisk","SSD","software","hardware","operating system","Windows",
                 "Mac","Linux","internet","wifi","program","coding","data","server"],

    "internet": ["online","wifi","koneksi","data","browsing","website","email","medsos",
                 "streaming","download","upload","bandwidth","kecepatan","provider",
                 "Telkom","Indosat","XL","Tri","Smartfren","cloud","server","hosting"],

    "aplikasi": ["hp","software","download","install","update","fitur","user","login",
                 "password","notifikasi","push notification","update","bug","error",
                 "Shopee","Tokopedia","Gojek","Grab","WhatsApp","Instagram","TikTok"],

    "robot": ["mesin","AI","otomasi","teknologi","android","humanoid","industri",
              "robot industri","robot medis","robot militer","drone","autonomous",
              "kecerdasan buatan","machine learning","sensor","aktuator","program"],

    "kamera": ["foto","gambar","lensa","resolusi","pixel","DSLR","mirrorless","action cam",
               "kamera hp","selfie","potret","fotografi","videografi","shutter","aperture",
               "ISO","tripod","flash","memori","SD card","foto landscape","portrait"],

    "telepon": ["hp","smartphone","menelepon","SMS","WA","komunikasi","nomor","sinyal",
                "iPhone","Samsung","Xiaomi","OPPO","Vivo","Realme","touchscreen",
                "baterai","charger","kamera hp","aplikasi","internet mobile"],

    # ══════════════════════════════════════════════════════════
    # SENI & BUDAYA
    # ══════════════════════════════════════════════════════════
    "musik": ["lagu","nada","melodi","ritme","harmoni","instrumen","vokal","penyanyi",
              "band","orkestra","jazz","pop","rock","dangdut","R&B","hip hop","EDM",
              "konser","album","rekaman","studio","mixing","mastering","produser"],

    "lagu": ["musik","nada","lirik","melodi","vokal","dinyanyikan","penyanyi","band",
             "lagu pop","lagu rock","lagu dangdut","lagu keroncong","lagu daerah",
             "hits","chart","streaming","Spotify","Apple Music","YouTube Music"],

    "tari": ["gerak","tubuh","irama","musik","penari","panggung","kostum","koreografi",
             "tari tradisional","tari modern","tari Jawa","tari Bali","tari Sunda",
             "tari Papua","tari Sumatera","tari Sulawesi","sendratari","ballet","modern dance"],

    "wayang": ["Jawa","tradisi","dalang","kulit","golek","pertunjukan","pewayangan",
               "Ramayana","Mahabharata","pandawa","kurawa","Arjuna","Bima","Werkudara",
               "Semar","Gareng","Petruk","Bagong","lakon","pagelaran","gamelan","suluk"],

    "batik": ["kain","motif","Jawa","tradisi","Indonesia","canting","malam","wax",
              "batik tulis","batik cap","batik printing","batik solo","batik yogya",
              "batik pekalongan","batik cirebon","batik madura","motif parang","mega mendung",
              "warisan budaya","UNESCO","kebudayaan","kain tradisional"],

    "gamelan": ["Jawa","Bali","alat musik","gong","kenong","saron","bonang","gender",
                "demung","slenthem","gambang","rebab","suling","kempul","kethuk",
                "karawitan","seniman","dalang","wayang","tari","tradisi","budaya"],

    "film": ["bioskop","layar","penonton","sutradara","aktor","aktris","skenario",
             "genre","horor","komedi","drama","action","romantis","animasi","dokumenter",
             "Indonesia","Hollywood","Bollywood","streaming","Netflix","Prime","Disney+"],

    "buku": ["membaca","tulisan","halaman","cerita","novel","fiksi","non-fiksi","penulis",
             "penerbit","ISBN","toko buku","gramedia","perpustakaan","e-book","audiobook",
             "buku pelajaran","buku anak","komik","manga","majalah","koran","ensiklopedia"],

    # ══════════════════════════════════════════════════════════
    # KERAJAAN & SEJARAH
    # ══════════════════════════════════════════════════════════
    "raja": ["ratu","mahkota","tahta","kerajaan","pedang","istana","rakyat","bangsawan","pangeran",
             "putri","sultan","kaisar","dinasti","perintah","kekuasaan","memerintah",
             "pahlawan","prajurit","perang","kemenangan","sejarah","majapahit","sriwijaya"],

    "ratu": ["raja","mahkota","istana","kerajaan","cantik","anggun","memerintah",
             "ratu elizabeth","ratu victoria","ratu cleopatra","ratu seba","ratu sheba",
             "putri mahkota","pasangan raja","keluarga kerajaan","monarki","dinasti"],

    "pangeran": ["raja","ratu","putri","mahkota","kerajaan","istana","pewaris","tahta",
                 "prince","pangeran tampan","dongeng","cerita","putra mahkota","hereditary",
                 "bangsawan","kastil","kuda putih","pahlawan","ksatria"],

    "putri": ["pangeran","raja","ratu","mahkota","kerajaan","istana","cantik","dongeng",
              "putri Cinderella","putri Rapunzel","putri salju","Snow White","Aurora",
              "putri mahkota","bangsawan","gaun","mahkota","pangeran tampan"],

    "sultan": ["kerajaan","istana","kerajaan Islam","kasultanan","keraton","Yogyakarta",
               "Surakarta","Banten","Aceh","Ternate","Tidore","Gowa","Bone","Kutai",
               "kebudayaan","tradisi","adat","upacara","wisata sejarah","heritage"],

    "kerajaan": ["raja","ratu","istana","mahkota","dinasti","sejarah","majapahit",
                 "sriwijaya","mataram","pajang","demak","banten","aceh","ternate",
                 "tidore","gowa","bone","kutai","tarumanegara","kediri","singasari",
                 "kerajaan Hindu","kerajaan Buddha","kerajaan Islam","penjajahan"],

    "pedang": ["senjata","tajam","baja","perang","ksatria","samurai","katana","keris",
               "golok","celurit","klewang","badik","rencong","mandau","parang",
               "tombak","panah","busur","perisai","pertempuran","pertarungan"],

    "mahkota": ["raja","ratu","emas","permata","berlian","kerajaan","tahta","kekuasaan",
                "mahkota kerajaan","upacara penobatan","simbol kekuasaan","heraldik",
                "crown","tiara","diadem","kemuliaan","kehormatan","lambang"],

    "tahta": ["raja","ratu","duduk","kekuasaan","kerajaan","istana","singgasana",
              "naik tahta","merebut tahta","pewaris tahta","mahkota","memerintah",
              "kekuasaan tertinggi","pemimpin","dominasi","otoritas","monarki"],

    # ══════════════════════════════════════════════════════════
    # ALAM SEMESTA
    # ══════════════════════════════════════════════════════════
    "planet": ["tata surya","bumi","mars","venus","merkurius","jupiter","saturnus",
               "uranus","neptunus","orbit","matahari","bulan","gravitasi","rotasi",
               "revolusi","atmosfer","cincin planet","sistem bintang","exoplanet"],

    "galaksi": ["bima sakti","bintang","semesta","teleskop","nebula","supernova",
                "lubang hitam","andromeda","spiral","elips","tak beraturan","galaksi lokal",
                "galaksi jauh","quasar","big bang","ekspansi","dark matter","dark energy"],

    "meteor": ["bintang jatuh","luar angkasa","batuan","tata surya","menghantam","bumi",
               "kawah","meteorit","asteroid","komet","Leonid","Perseid","Geminid",
               "hujan meteor","fenomena langit","shooting star","pijar","terbakar"],

    "bumi": ["planet","tanah","laut","udara","langit","matahari","gravitasi","atmosfer",
             "manusia","ekosistem","alam","lingkungan","hijau","biru","air","benua","samudra",
             "rotasi","revolusi","kutub","khatulistiwa","bulan","tata surya","bola bumi",
             "globe","peta","benua","asia","eropa","afrika","amerika","australia","antartika",
             "inti bumi","mantel bumi","kerak bumi","tektonik lempeng","gempa bumi"],

    "komet": ["ekor","es","debu","tata surya","Halley","Hale-Bopp","nucleus","coma",
              "mengorbit","matahari","periode","langka","fenomena langit","bersinar",
              "bintang berekor","frozen snowball","dirty snowball","astronomi"],

    # ══════════════════════════════════════════════════════════
    # KONSEP ABSTRAK
    # ══════════════════════════════════════════════════════════
    "mimpi": ["tidur","malam","angan","harapan","cita-cita","imajinasi","alam bawah sadar",
              "tidur nyenyak","bermimpi","mimpi indah","mimpi buruk","nightmare","lucid dream",
              "psikologi","Freud","Jung","mimpi jadi kenyataan","impian","visi"],

    "waktu": ["jam","menit","detik","hari","minggu","bulan","tahun","abad","era","masa",
              "sejarah","masa lalu","masa kini","masa depan","berlalu","berharga",
              "tidak bisa diulang","manajemen waktu","produktif","efisien","deadline"],

    "uang": ["rupiah","dolar","harga","beli","jual","ekonomi","modal","untung","rugi",
             "menabung","investasi","saham","obligasi","deposito","rekening","bank",
             "ATM","transfer","belanja","kemiskinan","kekayaan","gaji","penghasilan"],

    "ilmu": ["pengetahuan","belajar","pendidikan","riset","penelitian","universitas",
             "ilmuwan","penemuan","inovasi","sains","teknologi","matematika","fisika",
             "kimia","biologi","kedokteran","hukum","ekonomi","filsafat","sejarah"],

    "damai": ["tenang","harmonis","sejahtera","perang","konflik","resolusi","perdamaian",
              "damai dunia","PBB","NATO","negosiasi","diplomasi","rekonsiliasi",
              "toleransi","kerukunan","bhineka","NKRI","persatuan","kesatuan"],

    "harapan": ["cita-cita","impian","optimis","masa depan","positif","yakin","percaya",
                "doa","usaha","mimpi","ambisi","tujuan","target","semangat","motivasi",
                "inspirasi","harapan hidup","hope","expectation","aspiration"],

    "kehidupan": ["hidup","makna","tujuan","perjalanan","lahir","mati","keluarga",
                  "pekerjaan","hubungan","kebahagiaan","kesulitan","pelajaran","pengalaman",
                  "filosofi","arti hidup","why","purpose","legacy","warisan"],

    # ══════════════════════════════════════════════════════════
    # INDONESIA KHUSUS
    # ══════════════════════════════════════════════════════════
    "sawah": ["padi","petani","irigasi","hijau","air","musim tanam","panen","traktor",
              "cangkul","bajak","subak","persawahan","lumbung","gabah","beras",
              "nasi","ketahanan pangan","swasembada","bulog","impor beras","harga beras"],

    "desa": ["kampung","pedesaan","petani","nelayan","tradisi","adat","kerukunan",
             "kepala desa","lurah","RT","RW","musyawarah","gotong royong","sederhana",
             "alam","tenang","segar","hijau","dekat alam","wisata desa","agrowisata"],

    "orangutan": ["kera besar","primata","hutan","Kalimantan","Sumatera","merah","oranye",
                  "punah","dilindungi","konservasi","WWF","BOSF","deforestasi","sawit",
                  "habitat rusak","peliharaan ilegal","rehabilitasi","alam liar"],

    "komodo": ["kadal","besar","Komodo","flores","NTT","berbisa","lidah bercabang",
               "predator","rusa","babi hutan","kerbau","hewan purba","dinosaurus kecil",
               "taman nasional Komodo","labuan bajo","pulau Komodo","punah","dilindungi"],

    # ══════════════════════════════════════════════════════════
    # WARNA
    # ══════════════════════════════════════════════════════════
    "merah": ["darah","api","mawar","cinta","berani","bahaya","berhenti","Indonesia",
              "merah putih","bendera","semangat","panas","tomat","cabai","stroberi",
              "bata","maroon","crimson","scarlet","ruby","magenta","merah muda"],

    "biru": ["langit","laut","tenang","sejuk","air","langit cerah","biru tua","biru muda",
             "navy","cobalt","azure","sapphire","cerulean","biru langit","biru laut",
             "jeans","denim","seragam polisi","tentara","kehidupan","bumi"],

    "kuning": ["matahari","emas","cerah","ceria","pisang","lemang","kunyit","jagung",
               "kuning emas","kuning gading","kuning cerah","golden","amber","lemon",
               "kuning kehijauan","peringatan","hati-hati","traffic light"],

    "hijau": ["alam","pohon","daun","segar","sehat","lingkungan","organik","alami",
              "hijau muda","hijau tua","hijau zamrud","emerald","lime","sage","olive",
              "hijau botol","mint","hijau army","Islam","masjid","surga"],

    "putih": ["bersih","suci","murni","salju","awan","terang","polos","sederhana",
              "putih gading","putih bersih","cahaya","sinar","kertas","kapas",
              "lily","melati","jasmim","pernikahan","perdamaian","bendera putih"],

    "hitam": ["gelap","malam","misteri","elegan","formal","kuat","tegas","mewah",
              "hitam pekat","hitam legam","arang","batu bara","tinta","kafan",
              "duka","berkabung","hitam manis","kopi hitam","dark","noir"],

    "oranye": ["jeruk","buah","segar","terang","cerah","energi","anturias","antusias",
               "oranye terang","oranye tua","amber","peach","terracotta","coral",
               "matahari terbenam","senja","api","hangatkan","ramah"],

    "ungu": ["lavender","anggrek","royal","mewah","misterius","kreatif","spiritual",
             "ungu muda","ungu tua","violet","indigo","plum","grape","mauve",
             "aura","spiritual","magis","kebangsawanan","royal purple"],

    # ══════════════════════════════════════════════════════════
    # SIFAT FISIK
    # ══════════════════════════════════════════════════════════
    "panas": ["api","matahari","suhu","tinggi","membakar","terbakar","terik","gerah",
              "panas sekali","panas membara","hangat","lahar","volcano","tropis",
              "musim panas","cuaca panas","demam","tubuh panas","suhu tubuh"],

    "dingin": ["salju","es","beku","sejuk","segar","dingin sekali","menggigil",
               "kedinginan","jaket","selimut","heater","gunung","malam","suhu rendah",
               "kulkas","freezer","es batu","minuman dingin","angin dingin"],

    "besar": ["raksasa","besar sekali","besar-besaran","ukuran besar","massive","huge",
              "luas","lebar","panjang","tinggi","berat","besar hati","besar kepala",
              "gajah","paus","raksasa","gigantis","dinosaurus","supersize"],

    "kecil": ["mini","tiny","mungil","kerdil","kecil sekali","sangat kecil","imut",
              "bayi","anak-anak","semut","bakteri","virus","atom","molekul","nano",
              "microba","kecil hati","rendah diri"],

    "cepat": ["kilat","secepat","sprint","berlari kencang","mobil balap","jet","sonic",
              "supersonic","hypersonic","cahaya","kilat","ekspres","gesit","lincah",
              "agile","ninja","cheetah","falcon","kecepatan tinggi","rekord dunia"],

    "lambat": ["pelan","santai","lemot","slow motion","siput","penyu","sloth","malas",
               "mengantuk","lelah","berjalan pelan","tua","traffic jam","macet",
               "kura-kura","bekicot","bergerak lambat","tidak terburu-buru"],

    "kuat": ["tenaga","otot","kekuatan","fisik","kuat sekali","sangat kuat","kuat hati",
             "kuat mental","tahan banting","tabah","tangguh","perkasa","gagah","kokoh",
             "tegar","gajah","banteng","singa","kuda","angkat besi","powerlifting"],

    "lemah": ["tidak bertenaga","lunglai","lemas","capek","lelah","sakit","kurang sehat",
              "tidak berdaya","butuh bantuan","lemah fisik","lemah mental","depresi",
              "anxiety","tidak percaya diri","minder","insecure","down","jatuh"],

    # ══════════════════════════════════════════════════════════
    # OLAHRAGA
    # ══════════════════════════════════════════════════════════
    "sepak bola": ["bola","lapangan","gol","tendang","pemain","kiper","wasit","liga",
                   "piala","champion","FIFA","World Cup","Euro","Copa America","Liga 1",
                   "Persija","Arema","Persib","Bhayangkara","PSSI","Timnas"],

    "basket": ["bola basket","ring","lapangan","dribble","shoot","slam dunk","tim",
               "NBA","NBL","FIBA","pemain","pelatih","Lebron","Jordan","Kobe","Curry"],

    "badminton": ["shuttlecock","raket","net","smes","Korea","China","BWF","Thomas cup",
                  "Uber cup","Sudirman cup","bulu tangkis","Taufik Hidayat","Kevin Sanjaya",
                  "Marcus Gideon","Greysia","Apriyani","Indonesia","olimpiade","emas"],

    "renang": ["kolam","air","gaya bebas","gaya punggung","gaya dada","gaya kupu",
               "butterfly","freestyle","backstroke","breaststroke","olimpiade","FINA",
               "Michael Phelps","Ryan Lochte","kolam 50m","kolam 25m","renang cepat"],

    "lari": ["berlari","sprint","maraton","jogging","track","atlet","sepatu lari",
             "lari pagi","lari malam","half marathon","ultra marathon","ironman",
             "Usain Bolt","rekor dunia","lintasan","kilometer","pace","cadence"],

    "tinju": ["boxing","pukulan","ring","wasit","babak","round","KO","TKO","body shot",
              "uppercut","jab","cross","hook","Muhammad Ali","Mike Tyson","Manny Pacquiao",
              "sarung tinju","melindungi","kepala","badan","strategi","ring tinju"],

    "silat": ["pencak silat","bela diri","Indonesia","gerak","serang","tangkis","elak",
              "golok","keris","toya","pedang","jurus","perguruan","ikatan pencak silat",
              "SEA Games","olimpiade","tradisi","budaya","kuda-kuda","silat Betawi"],

    # ══════════════════════════════════════════════════════════
    # KELUARGA
    # ══════════════════════════════════════════════════════════
    "ayah": ["bapak","papa","abah","bapa","orang tua","keluarga","menafkahi","bekerja",
             "mendidik","melindungi","keras","tegas","penyayang","ayah kandung",
             "ayah tiri","ayah angkat","single parent","kewajiban ayah","peran ayah"],

    "ibu": ["mama","emak","bunda","mami","orang tua","keluarga","memasak","mengurus",
            "menyusui","melahirkan","penyayang","lembut","pengorbanan","ibu kandung",
            "ibu tiri","ibu angkat","single parent","hari ibu","kewajiban ibu"],

    "kakak": ["abang","kak","mbak","adik","saudara","keluarga","lebih tua","menjaga",
              "contoh","panutan","kakak laki","kakak perempuan","kakak kandung","kakak ipar"],

    "adik": ["dede","yunger","kakak","saudara","keluarga","lebih muda","disayang",
             "dijaga","adik laki","adik perempuan","adik kandung","adik ipar","bungsu"],

    "kakek": ["nenek","tua","tua sekali","bijaksana","pengalaman","cerita","dongeng",
              "kakek kandung","kakek dari ayah","kakek dari ibu","opa","abuya","buyut"],

    "nenek": ["kakek","tua","bijaksana","memasak","resep","dongeng","nenek kandung",
              "oma","omah","nenek sihir","nenek buyut","cerita","pengalaman hidup"],

    "suami": ["istri","menikah","pasangan","rumah tangga","nafkah","suami kandung",
              "pernikahan","keluarga","anak","rumah","setia","bersama","menafkahi"],

    "istri": ["suami","menikah","pasangan","rumah tangga","ibu rumah tangga","setia",
              "cinta","pernikahan","keluarga","anak","rumah","masak","mengurus"],

    "anak": ["keturunan","lahir","bayi","kecil","lucu","orang tua","keluarga","tumbuh",
             "berkembang","pendidikan","anak laki","anak perempuan","sulung","bungsu",
             "tengah","tunggal","yatim","piatu","anak angkat","anak kandung"],

    "teman": ["sahabat","kawan","bersama","bermain","ngobrol","cerita","berbagi",
              "teman dekat","teman jauh","teman baik","teman lama","teman baru",
              "pertemanan","persahabatan","setia","dipercaya","bermain bersama"],

    # ══════════════════════════════════════════════════════════
    # PERANG & SEJARAH
    # ══════════════════════════════════════════════════════════
    "perang": ["senjata","tentara","bom","pertempuran","kemenangan","kekalahan","korban",
               "perang dunia","perang dingin","perang saudara","perang gerilya","penjajahan",
               "kemerdekaan","revolusi","gencatan senjata","perdamaian","PBB","NATO"],

    "pahlawan": ["perjuangan","kemerdekaan","bela negara","pengorbanan","patriot","nasionalis",
                 "Soekarno","Hatta","Cut Nyak Dien","Kartini","Diponegoro","Hasanuddin",
                 "Ahmad Yani","Sudirman","Pattimura","Teuku Umar","pahlawan nasional"],

    "benteng": ["pertahanan","dinding","tebal","batu","perang","kerajaan","penjajah",
                "benteng Rotterdam","benteng Fort de Kock","benteng Vredeburg",
                "benteng Marlborough","sejarah","warisan budaya","museum","wisata"],

    "prajurit": ["tentara","berseragam","senjata","berperang","bela negara","barak",
                 "latihan","disiplin","pangkat","kopral","sersan","letnan","kapten",
                 "loyal","setia","siap tempur","siap berkorban","patriot"],
}

# ============================================================
# CATEGORY_MAP - Untuk boost hypernym/hyponym
# Jika kata_rahasia adalah anggota kategori, kata kategori ikut di-boost
# ============================================================
CATEGORY_MAP = {
    # Hewan
    "hewan":    ["harimau","singa","gajah","monyet","rusa","kuda","sapi","kambing","ayam",
                 "bebek","ular","buaya","katak","tikus","kelinci","kucing","anjing","rubah",
                 "beruang","panda","zebra","jerapah","lumba","paus","hiu","elang"],
    "binatang": ["harimau","singa","gajah","monyet","rusa","kuda","sapi","kambing","ayam",
                 "bebek","ular","buaya","katak","tikus","kelinci","kucing","anjing","rubah"],
    "satwa":    ["harimau","singa","gajah","monyet","rusa","orangutan","komodo","lumba","paus"],
    "mamalia":  ["harimau","singa","gajah","monyet","rusa","kuda","sapi","kambing","lumba","paus"],
    "predator": ["harimau","singa","hiu","elang","buaya","ular","beruang","rubah"],
    "reptil":   ["ular","buaya","komodo","katak","biawak","kadal","bunglon","penyu"],
    "serangga": ["lebah","kupu","semut","kumbang","capung","belalang","nyamuk","lalat"],
    "hewan laut":["ikan","lumba","paus","hiu","udang","kepiting","penyu","cumi","elang laut"],
    "burung":   ["elang","merpati","kakaktua","nuri","merak","jalak","kenari","kutilang"],
    "hewan peliharaan": ["kucing","anjing","kelinci","hamster","ikan hias","burung"],

    # Tumbuhan
    "tumbuhan": ["mawar","melati","anggrek","bambu","jati","pinus","cemara","beringin",
                 "mangga","rambutan","durian","pisang","pepaya","kelapa","padi","jagung"],
    "bunga":    ["mawar","melati","anggrek","kamboja","dahlia","matahari","lavender","lily"],
    "pohon":    ["bambu","jati","pinus","cemara","beringin","mangga","kelapa","pohon pisang"],
    "buah":     ["mangga","rambutan","durian","pisang","pepaya","salak","manggis","alpukat",
                 "nanas","semangka","jeruk","apel","anggur","strawberry","melon"],
    "sayuran":  ["bayam","kangkung","wortel","cabai","tomat","bawang","kentang","singkong","jagung"],
    "rempah":   ["jahe","kunyit","lengkuas","serai","bawang putih","bawang merah","cabai","lada","kayu manis"],

    # Makanan
    "makanan":  ["nasi","sate","bakso","soto","rendang","tempe","tahu","mie","roti","kue"],
    "minuman":  ["air","susu","kopi","teh","jus","sirup","es","cendol"],
    "masakan Indonesia": ["nasi","sate","bakso","soto","rendang","tempe","tahu","gado","ketoprak"],
    "bumbu":    ["garam","gula","merica","bawang","cabai","sambal","kecap","jahe","kunyit"],

    # Profesi
    "profesi":  ["dokter","guru","polisi","tentara","petani","nelayan","pedagang","pengacara"],
    "pekerjaan":["dokter","guru","polisi","tentara","petani","nelayan","pedagang","arsitek","pilot"],
    "pahlawan": ["tentara","petani","nelayan","guru","pejuang","patriot"],

    # Tubuh
    "tubuh":    ["kepala","mata","hidung","mulut","telinga","tangan","kaki","jantung","paru","otak"],
    "anggota tubuh": ["kepala","mata","hidung","mulut","telinga","tangan","kaki","perut","dada"],
    "organ":    ["jantung","paru","otak","hati","ginjal","lambung","usus","tulang"],

    # Perasaan
    "perasaan": ["cinta","rindu","sedih","bahagia","marah","takut","bangga","malu","senang"],
    "emosi":    ["cinta","rindu","sedih","bahagia","marah","takut","bangga","malu","senang"],
    "positif":  ["bahagia","senang","bangga","cinta","harapan","damai","syukur"],
    "negatif":  ["sedih","marah","takut","malu","rindu","kecewa","frustrasi"],

    # Tempat
    "bangunan": ["rumah","sekolah","masjid","gereja","istana","pasar","kantor","hotel"],
    "tempat ibadah": ["masjid","gereja","pura","vihara","klenteng","gereja Katolik"],
    "tempat wisata": ["pantai","gunung","danau","taman","museum","candi","benteng","istana"],

    # Transportasi
    "kendaraan": ["mobil","motor","kapal","pesawat","kereta","sepeda","perahu","bus","taksi"],
    "angkutan":  ["bus","taksi","ojek","angkot","kereta","becak","perahu","kapal"],

    # Alam
    "alam":     ["hutan","gunung","laut","sungai","pantai","danau","padang","sawah"],
    "cuaca":    ["hujan","angin","badai","petir","kilat","mendung","cerah","kabut","embun","salju"],
    "bencana":  ["banjir","gempa","gunung meletus","tsunami","longsor","kekeringan","badai"],
    "alam semesta": ["matahari","bulan","bintang","planet","galaksi","meteor","komet"],

    # Warna
    "warna":    ["merah","biru","kuning","hijau","putih","hitam","oranye","ungu","pink","coklat"],
    "warna primer": ["merah","biru","kuning"],
    "warna gelap": ["hitam","ungu","biru tua","maroon","abu gelap"],
    "warna terang": ["putih","kuning","pink","oranye","hijau muda"],

    # Sifat
    "sifat":    ["panas","dingin","besar","kecil","cepat","lambat","kuat","lemah","keras","lembut"],
    "ukuran":   ["besar","kecil","panjang","pendek","tinggi","rendah","lebar","sempit","tebal","tipis"],

    # Kerajaan
    "keluarga kerajaan": ["raja","ratu","pangeran","putri","sultan","mahkota","tahta"],
    "pemimpin": ["raja","ratu","sultan","kaisar","presiden","menteri","gubernur","walikota"],
    "senjata":  ["pedang","panah","tombak","keris","golok","mandau","rencong","senapan"],

    # Olahraga
    "olahraga": ["sepak bola","basket","badminton","renang","lari","tinju","silat","voli"],
    "olahraga air": ["renang","selancar","berlayar","kayak","polo air","selam"],

    # Keluarga
    "keluarga": ["ayah","ibu","kakak","adik","kakek","nenek","suami","istri","anak"],
    "orang tua": ["ayah","ibu"],
    "anak-anak": ["anak","adik","kakak"],
    "lansia":   ["kakek","nenek"],
}

# ============================================================
# MODEL WORD2VEC (fastText Wikipedia Indonesia)
# Sebagai fallback untuk kata yang tidak ada di PRECOMPUTED_RELATIONS
# ============================================================
MODEL_URL    = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.id.vec.gz"
MODEL_PATH   = "/tmp/wiki.id.vec"
MAX_WORDS    = 50000

def load_model_from_text(filepath):
    global word_vectors, vocab_list
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if count >= MAX_WORDS:
                break
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    word_vectors[word] = vec / norm
                    count += 1
            except:
                continue
    vocab_list = list(word_vectors.keys())
    print(f"[MODEL] Loaded {len(word_vectors)} kata")

def download_and_load_model():
    global model_ready, model_error
    try:
        if os.path.exists(MODEL_PATH):
            print("[MODEL] Loading dari disk...")
            load_model_from_text(MODEL_PATH)
        else:
            print("[MODEL] Downloading wiki.id.vec...")
            gz_path = MODEL_PATH + ".gz"
            urllib.request.urlretrieve(MODEL_URL, gz_path)
            print("[MODEL] Extracting...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(MODEL_PATH, 'wb') as f_out:
                    while True:
                        chunk = f_in.read(65536)
                        if not chunk:
                            break
                        f_out.write(chunk)
            os.remove(gz_path)
            load_model_from_text(MODEL_PATH)

        global KATA_LAYAK
        KATA_LAYAK = [k for k in KATA_LAYAK if k in word_vectors or k in PRECOMPUTED_RELATIONS]
        print(f"[MODEL] KATA_LAYAK valid: {len(KATA_LAYAK)}")
        model_ready = True
        print("[MODEL] ✅ Siap!")
    except Exception as e:
        model_error = str(e)
        print(f"[MODEL] ERROR: {e}")
        # Tetap ready pakai PRECOMPUTED_RELATIONS saja
        model_ready = True

import urllib.request
threading.Thread(target=download_and_load_model, daemon=True).start()

# ============================================================
# ALGORITMA RANKING
# 3 layer: PRECOMPUTED → CATEGORY_MAP → Word2Vec
# ============================================================
def hitung_ranking(kata_rahasia):
    kata_l = kata_rahasia.lower().strip()
    ranking = {kata_l: 1}
    rank    = 2

    # ── LAYER 1: PRECOMPUTED_RELATIONS ─────────────────────
    # Kata-kata yang sudah di-hardcode dengan relasi manualnya
    if kata_l in PRECOMPUTED_RELATIONS:
        related = PRECOMPUTED_RELATIONS[kata_l]
        for i, w in enumerate(related):
            wl = w.lower()
            if wl not in ranking:
                ranking[wl] = rank
                rank += 1

    rank_after_precomputed = rank

    # ── LAYER 2: CATEGORY_MAP ──────────────────────────────
    # Jika kata_rahasia adalah anggota kategori, boost kata kategorinya
    cat_boost = {}
    sib_boost = {}

    for cat_word, members in CATEGORY_MAP.items():
        if kata_l in members:
            # Boost kata kategori (hypernym)
            if cat_word not in ranking:
                cat_rank = rank_after_precomputed + len(cat_boost) * 2
                cat_boost[cat_word] = cat_rank

            # Boost sibling members (kata lain dalam kategori sama)
            for j, sib in enumerate(members):
                if sib != kata_l and sib not in ranking:
                    sib_rank = rank_after_precomputed + len(CATEGORY_MAP) * 3 + j * 2
                    if sib not in sib_boost or sib_boost[sib] > sib_rank:
                        sib_boost[sib] = sib_rank

    for w, r in sorted(cat_boost.items(), key=lambda x: x[1]):
        if w not in ranking:
            ranking[w] = r

    for w, r in sorted(sib_boost.items(), key=lambda x: x[1]):
        if w not in ranking:
            ranking[w] = r

    rank = max(ranking.values()) + 1

    # ── LAYER 3: WORD2VEC ──────────────────────────────────
    # Fallback untuk kata-kata yang belum ada di ranking
    if kata_l in word_vectors:
        target_vec = word_vectors[kata_l]
        words_arr  = np.array([word_vectors[w] for w in vocab_list], dtype=np.float32)
        sims       = words_arr.dot(target_vec)
        sorted_idx = np.argsort(-sims)

        for idx in sorted_idx:
            w = vocab_list[idx]
            if w == kata_l or w in ranking:
                continue
            if not w.isalpha() or len(w) < 2:
                continue
            if not any(c in "aiueo" for c in w):
                continue
            ranking[w] = rank
            rank += 1
            if rank > 12001:
                break

    return ranking

# ============================================================
# ENDPOINTS
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", "model_ready": model_ready,
        "model_error": model_error,
        "vocab_size": len(word_vectors),
        "kata_layak": len(KATA_LAYAK),
        "precomputed": len(PRECOMPUTED_RELATIONS),
        "categories": len(CATEGORY_MAP),
        "cache": len(proxy_ranking_cache),
    })

@app.route("/generate-word", methods=["POST"])
def generate_word():
    if not model_ready:
        return jsonify({"error": "Model belum siap"}), 503
    import random, time
    random.seed(int(time.time() * 1000) % 2**32)
    available = [k for k in KATA_LAYAK if k not in used_words]
    if len(available) < len(KATA_LAYAK) * 0.2:
        used_words.clear()
        available = list(KATA_LAYAK)
    random.shuffle(available)
    kata = available[0] if available else "laut"
    used_words.add(kata)
    return jsonify({"success": True, "kata": kata.upper()})

@app.route("/generate-ranking", methods=["POST"])
def generate_ranking():
    if not model_ready:
        return jsonify({"error": "Model belum siap"}), 503
    data = request.get_json() or {}
    kata_rahasia = (data.get("kata_rahasia") or "").lower().strip()
    if not kata_rahasia:
        return jsonify({"error": "kata_rahasia diperlukan"}), 400

    with cache_lock:
        if kata_rahasia in proxy_ranking_cache:
            r = proxy_ranking_cache[kata_rahasia]
            return jsonify({"success": True, "kata_rahasia": kata_rahasia,
                            "ranking": r, "jumlah": len(r), "method": "cache"})
    t0 = time.time()
    try:
        ranking = hitung_ranking(kata_rahasia)
        elapsed = time.time() - t0
        method = "precomputed+w2v" if kata_rahasia in PRECOMPUTED_RELATIONS else "w2v"
        print(f"[RANKING] '{kata_rahasia}' → {len(ranking)} kata | {elapsed:.2f}s | {method}")

        with cache_lock:
            if len(proxy_ranking_cache) >= 150:
                del proxy_ranking_cache[next(iter(proxy_ranking_cache))]
            proxy_ranking_cache[kata_rahasia] = ranking

        return jsonify({"success": True, "kata_rahasia": kata_rahasia,
                        "ranking": ranking, "jumlah": len(ranking), "method": method})
    except Exception as e:
        print(f"[RANKING] ERROR: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/check-word", methods=["POST"])
def check_word():
    data = request.get_json() or {}
    kata = (data.get("kata") or "").lower().strip()
    if not kata or len(kata) < 2:
        return jsonify({"valid": False, "reason": "Kata terlalu pendek"})
    if not kata.isalpha():
        return jsonify({"valid": False, "reason": "Hanya huruf A-Z"})
    if not any(c in "aiueo" for c in kata):
        return jsonify({"valid": False, "reason": "Bukan kata Indonesia"})
    # Cek di PRECOMPUTED atau Word2Vec
    in_precomputed = any(kata in rels for rels in PRECOMPUTED_RELATIONS.values())
    in_wv = kata in word_vectors
    in_kl = kata in KATA_LAYAK
    is_valid = in_precomputed or in_wv or in_kl or len(kata) >= 3
    return jsonify({"valid": is_valid})

@app.route("/", methods=["GET"])
def index():
    return jsonify({"app": "KONTEKS Proxy v8",
                    "model_ready": model_ready,
                    "precomputed_words": len(PRECOMPUTED_RELATIONS),
                    "categories": len(CATEGORY_MAP)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
