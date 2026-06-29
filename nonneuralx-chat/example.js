/**
 * example.js - command-line demo of NonNeuralX agent
 *
 * Run:
 *   node example.js
 *
 * Or with a custom corpus file:
 *   node example.js path/to/corpus.txt
 */

import { readFileSync } from "node:fs";
import { CORPORA, NonNeuralAgent } from "./src/index.js";

const corpusArg = process.argv[2];
const corpus = corpusArg ? readFileSync(corpusArg, "utf-8") : CORPORA.ml.text;

console.log("\n=== NonNeuralX agent ===\n");

const agent = new NonNeuralAgent({ order: 9, temperature: 0.9 });
const info = agent.learn(corpus);

console.log(`trained on ${info.chars.toLocaleString()} chars`);
console.log(`transitions: ${info.transitions.toLocaleString()}`);
console.log(`top keywords: ${info.keywords.join(", ")}\n`);

const questions = [
  "What is entropy?",
  "How does regularization work?",
  "Why use non-neural models?",
  "tasks: ingest(2)[cpu], validate(1)[cpu], deploy(1)[ops]\nresources: cpu=1, ops=1\nconstraints: ingest -> validate; deploy => validate",
];

for (const q of questions) {
  const { answer, stats, retrieved, intent, verification, plan } = agent.think(q, { maxLength: 250 });
  console.log("Q:", q);
  console.log("intent:", intent.taskType, "| entities:", intent.entities.join(", ") || "-");
  console.log("A:", answer);
  console.log("retrieved:", ((retrieved[0] || "") + "").slice(0, 60) + "...");
  console.log("verification:", verification.ok ? "ok" : `issues=${verification.issues.join("; ")}`);
  console.log("plan:", plan.steps.map((step) => step.action).join(" -> "));
  console.log("stats:", stats);
  console.log("---\n");
}
