const express = require("express");
const app = express();
app.use(express.json());

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const ROBLOX_SECRET  = process.env.ROBLOX_SECRET || "konteks-rahasia-2024";

app.post("/generate-ranking", async (req, res) => {
  if (req.headers["x-roblox-secret"] !== ROBLOX_SECRET) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  const { kata_rahasia } = req.body;
  if (!kata_rahasia) return res.status(400).json({ error: "kata_rahasia diperlukan" });

  const prompt = `Kamu adalah sistem ranking kata untuk game tebak kata bahasa Indonesia mirip Contexto.
Kata rahasia: "${kata_rahasia.toUpperCase()}"
Buat daftar 300 kata bahasa Indonesia yang paling berkaitan, dari PALING MIRIP (ranking 1) ke PALING JAUH (ranking 300+).
Panduan:
- Ranking 1-5: Sinonim langsung
- Ranking 6-20: Sangat berkaitan
- Ranking 21-80: Berkaitan tidak langsung
- Ranking 81-200: Sedikit berkaitan
- Ranking 201-500: Sangat jauh
Format output HANYA: kata=ranking (satu per baris, tanpa penjelasan apapun)
Contoh:
minum=2
sungai=8
laut=9`;

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 3000 },
      }),
    });
    if (!response.ok) throw new Error(`Gemini error ${response.status}: ${await response.text()}`);
    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const ranking = {};
    for (const line of text.split("\n")) {
      const match = line.trim().match(/^([a-zA-Z\s\-']+)=(\d+)$/);
      if (match) {
        const kata = match[1].trim().toLowerCase();
        const rank = parseInt(match[2]);
        if (kata.length >= 2 && rank > 0) ranking[kata] = rank;
      }
    }
    console.log(`[OK] "${kata_rahasia}" => ${Object.keys(ranking).length} kata`);
    res.json({ success: true, kata_rahasia, ranking });
  } catch (err) {
    console.error(`[ERROR]`, err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get("/health", (_, res) => res.json({ status: "ok", gemini: !!GEMINI_API_KEY }));
app.listen(process.env.PORT || 3000, () => console.log("KONTEKS Proxy running!"));
