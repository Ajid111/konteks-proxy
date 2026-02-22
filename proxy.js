// ============================================================
// KONTEKS Proxy Server v3
// AI generate KATA RAHASIA + ranking hingga ribuan otomatis
// ============================================================

const express = require("express");
const app = express();
app.use(express.json());

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const ROBLOX_SECRET  = process.env.ROBLOX_SECRET || "konteks-rahasia-2024";

// Kata yang sudah pernah dipakai (reset kalau server restart)
const usedWords = new Set();

// ============================================================
// ENDPOINT 1: Generate kata rahasia baru (random dari AI)
// ============================================================
app.post("/generate-word", async (req, res) => {
  if (req.headers["x-roblox-secret"] !== ROBLOX_SECRET) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const prompt = `Kamu adalah generator kata untuk game tebak kata bahasa Indonesia.

Tugasmu: Pilih SATU kata bahasa Indonesia yang menarik untuk dijadikan kata rahasia dalam game.

Syarat kata:
- Kata benda umum yang dikenal semua orang Indonesia
- Bukan nama orang/tempat/merek
- Tidak terlalu mudah, tidak terlalu susah
- Bervariasi: bisa benda, alam, makanan, hewan, konsep, dll
- Kata sudah dipakai (jangan pilih ini): ${Array.from(usedWords).join(", ") || "belum ada"}

Contoh kata yang bagus: LAUT, MIMPI, WAKTU, CAHAYA, ANGIN, NASI, HARIMAU, GELAP, SUNGAI, PAHLAWAN

Balas HANYA dengan satu kata, huruf kapital semua, tanpa penjelasan apapun.
Contoh: LAUT`;

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.9, maxOutputTokens: 20 },
      }),
    });

    if (!response.ok) throw new Error(`Gemini error ${response.status}`);
    const data = await response.json();
    const kata = (data.candidates?.[0]?.content?.parts?.[0]?.text || "")
      .trim()
      .toUpperCase()
      .replace(/[^A-Z]/g, "");

    if (!kata) throw new Error("Kata kosong dari AI");

    usedWords.add(kata);
    if (usedWords.size > 200) {
      // Reset setelah 200 kata agar bisa dipakai lagi
      const arr = Array.from(usedWords);
      usedWords.clear();
      arr.slice(-50).forEach(w => usedWords.add(w)); // simpan 50 terakhir
    }

    console.log(`[WORD] Generated: ${kata} (total used: ${usedWords.size})`);
    res.json({ success: true, kata });

  } catch (err) {
    console.error("[WORD ERROR]", err.message);
    // Fallback: pilih kata random dari daftar cadangan
    const fallback = ["LAUT","MIMPI","WAKTU","CAHAYA","ANGIN","NASI","HARIMAU",
      "GELAP","SUNGAI","PAHLAWAN","HUJAN","GUNUNG","BULAN","BINTANG","API",
      "POHON","AWAN","PASIR","DAUN","BATU","PELANGI","KABUT","EMBUN","BADAI"];
    const kata = fallback[Math.floor(Math.random() * fallback.length)];
    res.json({ success: true, kata, fallback: true });
  }
});

// ============================================================
// ENDPOINT 2: Generate ranking untuk satu kata (hingga ribuan)
// ============================================================
app.post("/generate-ranking", async (req, res) => {
  if (req.headers["x-roblox-secret"] !== ROBLOX_SECRET) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { kata_rahasia } = req.body;
  if (!kata_rahasia) return res.status(400).json({ error: "kata_rahasia diperlukan" });

  const prompt = `Kamu adalah sistem AI untuk game tebak kata bahasa Indonesia bernama KONTEKS, mirip dengan game Contexto.

Kata rahasia: "${kata_rahasia}"

Tugasmu adalah membuat daftar kata bahasa Indonesia dengan sistem ranking kedekatan makna/konteks.

PENTING - Cara kerja ranking yang BENAR:
- Ranking 1: Kata itu sendiri
- Ranking 2-10: Sinonim LANGSUNG atau kata yang sangat identik maknanya
- Ranking 11-50: Kata yang SANGAT berkaitan (benda terkait, fungsi sama, sering muncul bersama)
- Ranking 51-200: Kata yang BERKAITAN (dalam konteks yang sama, satu kategori)
- Ranking 201-500: Kata yang SEDIKIT berkaitan (hubungan tidak langsung, konteks lebih luas)
- Ranking 501-1000: Kata yang JAUH tapi masih ada benang merah
- Ranking 1001-3000: Kata yang SANGAT JAUH tapi masih bisa dihubungkan dengan logika

Contoh untuk kata "LAUT":
ombak=3, samudra=5, nelayan=12, kapal=15, ikan=18, pantai=20, asin=25, biru=40,
sungai=80, danau=90, hujan=150, angin=200, gunung=400, hutan=600, kota=900, mobil=1500

Buat 500 kata dengan distribusi:
- 10 kata ranking 1-10
- 40 kata ranking 11-50  
- 150 kata ranking 51-200
- 150 kata ranking 201-500
- 100 kata ranking 501-1000
- 50 kata ranking 1001-3000

Format output HANYA baris-baris: kata=ranking
Tanpa penjelasan, tanpa komentar, langsung daftar saja.`;

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 8192,
        },
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Gemini error ${response.status}: ${errText}`);
    }

    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";

    // Parse kata=ranking
    const ranking = {};
    for (const line of text.split("\n")) {
      const match = line.trim().match(/^([a-zA-Z\s\-']+)=(\d+)$/);
      if (match) {
        const kata = match[1].trim().toLowerCase();
        const rank = parseInt(match[2]);
        if (kata.length >= 2 && rank > 0 && rank <= 9999) {
          ranking[kata] = rank;
        }
      }
    }

    const jumlah = Object.keys(ranking).length;
    console.log(`[RANKING] "${kata_rahasia}" => ${jumlah} kata`);
    res.json({ success: true, kata_rahasia, ranking, jumlah });

  } catch (err) {
    console.error(`[RANKING ERROR] ${kata_rahasia}:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get("/health", (_, res) => {
  res.json({ status: "ok", gemini: !!GEMINI_API_KEY, used_words: usedWords.size });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`KONTEKS Proxy v3 running on port ${PORT}`);
  console.log(`API Key: ${GEMINI_API_KEY ? "SET" : "MISSING!"}`);
});
