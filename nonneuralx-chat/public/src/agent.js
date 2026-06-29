import { HierarchicalMemory } from "./memory.js";
import { CharacterMarkovGenerator } from "./markov.js";

/**
 * NonNeuralAgent
 *
 * Pipeline per query:
 * 1) Retrieve relevant long-term + short-term chunks
 * 2) Build prompt: Context + Question + Answer
 * 3) Generate with character-level Markov chain
 * 4) Return { query, answer, retrieved, prompt, stats }
 */
export class NonNeuralAgent {
  constructor(opts = {}) {
    this.order = opts.order ?? 9;
    this.temperature = opts.temperature ?? 0.9;
    this.maxLength = opts.maxLength ?? 400;
    this.topK = opts.topK ?? 3;

    this.memory = new HierarchicalMemory();
    this.markov = new CharacterMarkovGenerator(this.order, this.temperature);
    this.trained = false;
  }

  learn(corpus) {
    const text = String(corpus || "");
    this.memory.addText(text);
    this.markov.fit(text);
    this.trained = true;

    return {
      chars: text.length,
      transitions: this.markov.transitionCount,
      chunks: this.memory.chunkCount,
      keywords: this.memory.topKeywords(8),
    };
  }

  think(query, opts = {}) {
    if (!this.trained) {
      throw new Error("Call learn(corpus) before think().");
    }

    const order = opts.order ?? this.order;
    const temperature = opts.temperature ?? this.temperature;
    const maxLength = opts.maxLength ?? this.maxLength;
    const topK = opts.topK ?? this.topK;

    this.markov.order = order;
    this.markov.temperature = temperature;

    const longCtx = this.memory.retrieve(query, topK, "long");
    const shortCtx = this.memory.retrieve(query, 2, "short");
    const context = [...longCtx, ...shortCtx].join(" ").slice(0, 500);

    const prompt = `Context: ${context} Question: ${query} Answer:`;
    const raw = this.markov.generate(prompt, maxLength);

    const answer = raw.replace(/^.*?Answer:\s*/s, "").replace(/^\s+/, "") || raw;

    return {
      query,
      answer: answer.slice(0, maxLength),
      retrieved: longCtx,
      prompt,
      stats: {
        order,
        temperature,
        answerLength: answer.length,
        transitions: this.markov.transitionCount,
      },
    };
  }

  get stats() {
    return {
      trained: this.trained,
      transitions: this.markov.transitionCount,
      chunks: this.memory.chunkCount,
      keywords: this.memory.topKeywords(10),
    };
  }
}
