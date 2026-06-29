/**
 * CharacterMarkovGenerator
 *
 * Order-N character-level Markov chain with temperature sampling.
 * Blends contexts from order 1 through N (higher orders preferred).
 */
export class CharacterMarkovGenerator {
  constructor(order = 9, temperature = 0.9) {
    this.order = order;
    this.temperature = temperature;
    this.trans = {};
    this.vocab = [];
  }

  fit(text) {
    text = String(text || "").replace(/\s+/g, " ").trim();
    if (!text) return this;

    this.vocab = [...new Set(text)];
    this.trans = {};

    for (let o = 1; o <= this.order; o++) {
      for (let i = 0; i < text.length - o; i++) {
        const ctx = text.slice(i, i + o);
        const nxt = text[i + o];

        if (!this.trans[ctx]) this.trans[ctx] = {};
        this.trans[ctx][nxt] = (this.trans[ctx][nxt] || 0) + 1;
      }
    }

    return this;
  }

  generate(prompt = "", maxLen = 300) {
    const gen = prompt ? [...prompt.slice(-this.order)] : [" "];

    for (let step = 0; step < maxLen; step++) {
      const probs = {};

      for (let o = Math.min(this.order, gen.length); o >= 1; o--) {
        const ctx = gen.slice(-o).join("");
        const counts = this.trans[ctx];
        if (!counts) continue;

        const total = Object.values(counts).reduce((s, v) => s + v, 0);
        for (const [ch, cnt] of Object.entries(counts)) {
          probs[ch] = (probs[ch] || 0) + cnt / total;
        }
      }

      if (!Object.keys(probs).length) {
        const fallback = this.vocab[Math.floor(Math.random() * this.vocab.length)] || " ";
        gen.push(fallback);
        continue;
      }

      const chars = Object.keys(probs);
      let ps = chars.map((c) => Math.pow(probs[c], 1 / Math.max(this.temperature, 1e-6)));
      const sum = ps.reduce((s, x) => s + x, 0) + 1e-12;
      ps = ps.map((x) => x / sum);

      let r = Math.random();
      let acc = 0;
      let chosen = chars[chars.length - 1];
      for (let i = 0; i < chars.length; i++) {
        acc += ps[i];
        if (r < acc) {
          chosen = chars[i];
          break;
        }
      }

      gen.push(chosen);

      if (".!?".includes(chosen) && gen.length > 50) break;
    }

    return gen.join("").replace(/\s+/g, " ").trim();
  }

  get transitionCount() {
    return Object.values(this.trans).reduce((s, v) => s + Object.keys(v).length, 0);
  }
}
