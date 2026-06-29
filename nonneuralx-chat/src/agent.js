import { HierarchicalMemory } from "./memory.js";
import { CharacterMarkovGenerator } from "./markov.js";
import { HybridSymbolicSystem } from "./symbolic.js";

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
    this.symbolic = new HybridSymbolicSystem({
      order: this.order,
      temperature: this.temperature,
      maxLength: this.maxLength,
      topK: this.topK,
      memory: this.memory,
      markov: this.markov,
    });
    this.trained = false;
  }

  learn(corpus) {
    const info = this.symbolic.learn(corpus);
    this.trained = true;
    return info;
  }

  think(query, opts = {}) {
    if (!this.trained) {
      throw new Error("Call learn(corpus) before think().");
    }
    return this.symbolic.think(query, {
      order: opts.order ?? this.order,
      temperature: opts.temperature ?? this.temperature,
      maxLength: opts.maxLength ?? this.maxLength,
      topK: opts.topK ?? this.topK,
    });
  }

  get stats() {
    return {
      trained: this.trained,
      transitions: this.markov.transitionCount,
      chunks: this.memory.chunkCount,
      facts: this.symbolic.graph.triples.length,
      keywords: this.memory.topKeywords(10),
    };
  }
}
