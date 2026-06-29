/**
 * HierarchicalMemory
 *
 * Three-tier memory system:
 * - short-term: most recent chunks (sliding window)
 * - long-term: full corpus history
 * - conceptual: keyword frequency index
 *
 * Retrieval uses 512-dim character-frequency cosine similarity.
 */
export class HierarchicalMemory {
  constructor() {
    this.shortTerm = [];
    this.longTerm = [];
    this.conceptual = {};
  }

  addText(text, chunkSize = 120) {
    text = String(text || "").replace(/\s+/g, " ").trim();
    if (!text) return;

    const step = Math.max(1, Math.floor(chunkSize / 2));
    const chunks = [];
    for (let i = 0; i < text.length; i += step) {
      const chunk = text.slice(i, i + chunkSize);
      if (chunk) chunks.push(chunk);
    }

    this.shortTerm = chunks.slice(-20);
    this.longTerm.push(...chunks);

    for (const chunk of chunks) {
      const words = chunk.toLowerCase().match(/\b\w{4,}\b/g) || [];
      for (const word of words) {
        this.conceptual[word] = (this.conceptual[word] || 0) + 1;
      }
    }
  }

  _vec(text) {
    const v = new Float32Array(512);
    for (const c of text) {
      v[c.charCodeAt(0) % 512] += 1;
    }
    let normSq = 0;
    for (let i = 0; i < 512; i++) normSq += v[i] * v[i];
    const norm = Math.sqrt(normSq) || 1;

    for (let i = 0; i < 512; i++) v[i] = v[i] / norm;
    return v;
  }

  _sim(a, b) {
    let s = 0;
    for (let i = 0; i < 512; i++) s += a[i] * b[i];
    return s;
  }

  retrieve(query, topK = 3, level = "long") {
    const pool = level === "short" ? this.shortTerm : this.longTerm;
    if (!pool.length) return [];

    const qv = this._vec(String(query || ""));
    const scored = pool.map((chunk) => ({
      chunk,
      score: this._sim(qv, this._vec(chunk)),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((x) => x.chunk);
  }

  topKeywords(n = 10) {
    return Object.entries(this.conceptual)
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([w]) => w);
  }

  get chunkCount() {
    return this.longTerm.length;
  }
}
