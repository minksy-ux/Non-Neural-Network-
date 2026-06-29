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

  _distributionForContext(context) {
    const probs = {};

    for (let o = Math.min(this.order, context.length); o >= 1; o--) {
      const ctx = context.slice(-o);
      const counts = this.trans[ctx];
      if (!counts) continue;

      const total = Object.values(counts).reduce((s, v) => s + v, 0);
      for (const [ch, cnt] of Object.entries(counts)) {
        probs[ch] = (probs[ch] || 0) + cnt / total;
      }
    }

    const chars = Object.keys(probs);
    if (!chars.length) return [];

    const temperature = Math.max(this.temperature, 1e-6);
    const weights = chars.map((ch) => Math.pow(probs[ch], 1 / temperature));
    const sum = weights.reduce((s, x) => s + x, 0) + 1e-12;

    return chars
      .map((ch, index) => ({ ch, prob: weights[index] / sum }))
      .sort((a, b) => b.prob - a.prob);
  }

  _sampleFromDistribution(distribution) {
    if (!distribution.length) return null;

    let r = Math.random();
    let acc = 0;
    for (const item of distribution) {
      acc += item.prob;
      if (r < acc) return item.ch;
    }
    return distribution[distribution.length - 1].ch;
  }

  generate(prompt = "", maxLen = 300) {
    const gen = prompt ? [...prompt.slice(-this.order)] : [" "];

    for (let step = 0; step < maxLen; step++) {
      const distribution = this._distributionForContext(gen.join(""));
      if (!distribution.length) {
        const fallback = this.vocab[Math.floor(Math.random() * this.vocab.length)] || " ";
        gen.push(fallback);
        continue;
      }

      const chosen = this._sampleFromDistribution(distribution);
      gen.push(chosen);

      if (".!?".includes(chosen) && gen.length > 50) break;
    }

    return gen.join("").replace(/\s+/g, " ").trim();
  }

  generateBest(prompt = "", maxLen = 300, beamWidth = 6) {
    const seed = prompt ? prompt.replace(/\s+/g, " ").trim().slice(-Math.max(1, this.order)) : " ";
    let beams = [{ text: seed, score: 0, ended: false }];

    for (let step = 0; step < maxLen; step++) {
      const nextBeams = [];

      for (const beam of beams) {
        if (beam.ended) {
          nextBeams.push(beam);
          continue;
        }

        const distribution = this._distributionForContext(beam.text);
        if (!distribution.length) {
          nextBeams.push({ ...beam, ended: true });
          continue;
        }

        for (const item of distribution.slice(0, beamWidth)) {
          const nextText = beam.text + item.ch;
          const punctuationBonus = ".!?".includes(item.ch) && nextText.length > 50 ? 0.15 : 0;
          nextBeams.push({
            text: nextText,
            score: beam.score + Math.log(item.prob + 1e-12) + punctuationBonus,
            ended: ".!?".includes(item.ch) && nextText.length > 50,
          });
        }
      }

      nextBeams.sort((a, b) => b.score - a.score || b.text.length - a.text.length);
      beams = nextBeams.slice(0, Math.max(beamWidth, 4));
      if (beams.every((beam) => beam.ended)) break;
    }

    const best = beams.reduce((winner, current) => {
      const winnerScore = winner.score / Math.pow(Math.max(winner.text.length, 1), 0.65);
      const currentScore = current.score / Math.pow(Math.max(current.text.length, 1), 0.65);
      return currentScore > winnerScore ? current : winner;
    }, beams[0]);

    return best.text.replace(/\s+/g, " ").trim();
  }

  get transitionCount() {
    return Object.values(this.trans).reduce((s, v) => s + Object.keys(v).length, 0);
  }
}
