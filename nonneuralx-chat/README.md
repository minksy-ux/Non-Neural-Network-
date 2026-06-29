# NonNeuralX Chat

Interactive browser playground for a non-neural chat pipeline:

- Hierarchical memory retrieval (short-term + long-term + conceptual index)
- Character-level Markov generation
- NonNeuralAgent response format: `{ query, answer, retrieved, prompt, stats }`
- Browser transcript persistence with export/import
- Seed corpus editor with one-click preset loading and retraining
- Pinned context persistence alongside chat history

## Structure

- `src/` importable JS modules
- `public/` browser UI with ES modules
- `offline.html` single-file browser demo
- `example.js` CLI demo

The browser UI stores chat history in `localStorage` and supports JSON export/import.
The `offline.html` file is a standalone, no-build demo you can open directly in a browser.

## Run

```bash
npm install
npm start
```

Open `http://localhost:3000`.

CLI demo:

```bash
npm run demo
```
