function normalizeText(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function keywords(text) {
  return [...new Set((normalizeText(text).toLowerCase().match(/\b\w{3,}\b/g) || []))];
}

function overlapScore(a, b) {
  const left = keywords(a);
  const right = new Set(keywords(b));
  if (!left.length || !right.size) return 0;
  const hits = left.reduce((count, word) => count + (right.has(word) ? 1 : 0), 0);
  return hits / left.length;
}

function uniqueBy(list, keyFn) {
  const seen = new Set();
  const result = [];
  for (const item of list) {
    const key = keyFn(item);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(item);
  }
  return result;
}

export class IntentParser {
  parse(text) {
    const raw = normalizeText(text);
    const lower = raw.toLowerCase();
    const tokens = keywords(raw);
    const dsl = this._taskDsl(text);
    const entities = this._entities(raw);
    const constraints = this._constraints(raw);
    const relations = this._relations(raw, dsl);
    const taskType = this._taskType(lower, constraints);

    const structuredQuery = {
      mode: taskType,
      focus: tokens.slice(0, 8),
      entities,
      constraints,
      relations,
      dsl,
      wantsEvidence: /why|how|what|which|cite|evidence|source|support|verify|prove|check/.test(lower),
      wantsPlanning: /plan|schedule|decompose|execute|steps?|workflow|pipeline|task graph/.test(lower),
      wantsReasoning: /infer|reason|deduce|derive|theorem|logic|proof|constraint|solve/.test(lower),
    };

    const graph = this._taskGraph(raw, taskType, entities, constraints, relations, dsl);

    return {
      raw,
      normalized: raw,
      taskType,
      intent: taskType,
      tokens,
      dsl,
      entities,
      constraints,
      relations,
      structuredQuery,
      graph,
      confidence: this._confidence(taskType, tokens, constraints, relations),
    };
  }

  _taskType(lower, constraints) {
    if (/\b(tasks?|constraints?|depends|dependency|schedule|plan)\s*:/i.test(lower)) return "planner";
    if (/\b(schedule|dependency|order|precedence|after|before|timeline)\b/.test(lower) || constraints.length) return "planner";
    if (/\b(prove|verify|check|contradiction|satisfy|satisfies|unsat|sat)\b/.test(lower)) return "verifier";
    if (/\b(compare|contrast|versus|vs\.?|tradeoff)\b/.test(lower)) return "comparison";
    if (/\b(what|define|definition|mean|means|is)\b/.test(lower)) return "retrieval";
    if (/\b(how|why|explain|because|reason|derive)\b/.test(lower)) return "reasoning";
    if (/\b(generate|write|draft|compose|synthesize)\b/.test(lower)) return "generation";
    return "analysis";
  }

  _entities(raw) {
    const stopwords = new Set(["what", "why", "how", "when", "where", "which", "who", "whom", "whose", "is", "are", "do", "does", "did", "can", "could", "should", "would", "please"]);
    const quoted = raw.match(/"([^"]+)"|'([^']+)'/g)?.map((s) => s.slice(1, -1)) || [];
    const proper = raw.match(/\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*/g) || [];
    return uniqueBy([...quoted, ...proper], (value) => value.toLowerCase())
      .filter((value) => !stopwords.has(value.toLowerCase()))
      .slice(0, 12);
  }

  _constraints(raw) {
    const matches = [];
    const patterns = [
      /\bmust\s+([^.;]+)/gi,
      /\bonly\s+([^.;]+)/gi,
      /\bwithout\s+([^.;]+)/gi,
      /\bbefore\s+([^.;]+)/gi,
      /\bafter\s+([^.;]+)/gi,
      /\bunless\s+([^.;]+)/gi,
      /\bif\s+([^.;]+)/gi,
    ];

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(raw))) {
        matches.push(match[1].trim());
      }
    }

    return uniqueBy(matches.filter(Boolean), (value) => value.toLowerCase()).slice(0, 10);
  }

  _taskDsl(rawText) {
    const raw = String(rawText || "");
    const lines = raw
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);

    const taskLine = lines.find((line) => /^tasks?\s*:/i.test(line)) || raw.match(/tasks?\s*:\s*([^\n]+)/i)?.[0] || "";
    const resourceLine = lines.find((line) => /^resources?\s*:/i.test(line)) || raw.match(/resources?\s*:\s*([^\n]+)/i)?.[0] || "";
    const constraintLine = lines.find((line) => /^constraints?\s*:/i.test(line)) || raw.match(/constraints?\s*:\s*([^\n]+)/i)?.[0] || "";
    const ruleLines = lines
      .filter((line) => !/^tasks?\s*:/i.test(line) && !/^resources?\s*:/i.test(line))
      .map((line) => line.replace(/^constraints?\s*:/i, "").trim());

    const taskSpecs = this._parseTaskList(taskLine.replace(/^tasks?\s*:/i, ""));
    const tasks = taskSpecs.map((task) => task.name);
    const resources = this._parseResourceList(resourceLine.replace(/^resources?\s*:/i, ""));
    const rawRuleText = [constraintLine.replace(/^constraints?\s*:/i, ""), ...ruleLines]
      .map((part) => part.trim())
      .filter(Boolean)
      .filter((part, index, list) => list.indexOf(part) === index)
      .join("; ");

    const rules = this._parseDslRules(rawRuleText);
    if (!tasks.length && !rules.length) return null;

    const derivedTasks = [];
    for (const rule of rules) {
      if (rule.from) derivedTasks.push(rule.from);
      if (rule.to) derivedTasks.push(rule.to);
    }

    return {
      tasks: uniqueBy([...tasks, ...derivedTasks].filter(Boolean), (value) => value.toLowerCase()),
      taskSpecs,
      resources,
      rules,
      raw: rawRuleText,
    };
  }

  _parseTaskList(text) {
    return uniqueBy(
      String(text || "")
        .split(/[,|]/)
        .map((item) => this._parseTaskSpec(item))
        .filter((value) => value?.name),
      (value) => value.name.toLowerCase(),
    );
  }

  _parseTaskSpec(text) {
    const raw = normalizeText(String(text || "").replace(/^[-*]\s*/, ""));
    if (!raw) return null;

    const match = raw.match(/^([^([\]]+?)(?:\((\d+)\))?(?:\[([^\]]+)\])?$/);
    if (!match) {
      return { name: raw, duration: 1, resource: null };
    }

    return {
      name: normalizeText(match[1]),
      duration: Math.max(1, Number(match[2] || 1)),
      resource: normalizeText(match[3] || "") || null,
    };
  }

  _parseResourceList(text) {
    const resources = {};
    const parts = String(text || "")
      .split(/[,|]/)
      .map((item) => normalizeText(item))
      .filter(Boolean);

    for (const part of parts) {
      const match = part.match(/^([^=]+?)\s*=\s*(\d+)$/);
      if (!match) continue;
      const name = normalizeText(match[1]);
      const capacity = Math.max(1, Number(match[2] || 1));
      resources[name] = capacity;
    }

    return resources;
  }

  _parseDslRules(text) {
    const rules = [];
    const segments = String(text || "")
      .split(/[;\n]+/)
      .flatMap((segment) => segment.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/))
      .map((segment) => normalizeText(segment))
      .filter(Boolean);

    for (const segment of segments) {
      let match = segment.match(/^(.+?)\s*->\s*(.+)$/);
      if (match) {
        rules.push({ relation: "before", from: normalizeText(match[1]), to: normalizeText(match[2]), source: segment });
        continue;
      }

      match = segment.match(/^(.+?)\s*=>\s*(.+)$/);
      if (match) {
        rules.push({ relation: "depends_on", from: normalizeText(match[1]), to: normalizeText(match[2]), source: segment });
        continue;
      }

      match = segment.match(/^(.+?)\s*!\s*(.+)$/);
      if (match) {
        rules.push({ relation: "mutex", from: normalizeText(match[1]), to: normalizeText(match[2]), source: segment });
        continue;
      }

      match = segment.match(/^(.+?)\s+before\s+(.+)$/i);
      if (match) {
        rules.push({ relation: "before", from: normalizeText(match[1]), to: normalizeText(match[2]), source: segment });
        continue;
      }

      match = segment.match(/^(.+?)\s+after\s+(.+)$/i);
      if (match) {
        rules.push({ relation: "after", from: normalizeText(match[1]), to: normalizeText(match[2]), source: segment });
        continue;
      }

      match = segment.match(/^(.+?)\s+(?:depends on|requires)\s+(.+)$/i);
      if (match) {
        rules.push({ relation: "depends_on", from: normalizeText(match[1]), to: normalizeText(match[2]), source: segment });
      }
    }

    return uniqueBy(rules, (value) => `${value.relation}:${value.from.toLowerCase()}:${value.to.toLowerCase()}`);
  }

  _relations(raw, dsl = null) {
    const relations = [];
    const patterns = [
      { relation: "before", regex: /\b([A-Za-z0-9_\- ]+?)\s+before\s+([A-Za-z0-9_\- ]+?)(?:[.;,]|$)/gi },
      { relation: "after", regex: /\b([A-Za-z0-9_\- ]+?)\s+after\s+([A-Za-z0-9_\- ]+?)(?:[.;,]|$)/gi },
      { relation: "depends_on", regex: /\b([A-Za-z0-9_\- ]+?)\s+depends? on\s+([A-Za-z0-9_\- ]+?)(?:[.;,]|$)/gi },
      { relation: "requires", regex: /\b([A-Za-z0-9_\- ]+?)\s+requires\s+([A-Za-z0-9_\- ]+?)(?:[.;,]|$)/gi },
      { relation: "uses", regex: /\b([A-Za-z0-9_\- ]+?)\s+uses\s+([A-Za-z0-9_\- ]+?)(?:[.;,]|$)/gi },
    ];

    for (const { relation, regex } of patterns) {
      let match;
      while ((match = regex.exec(raw))) {
        relations.push({ relation, from: match[1].trim(), to: match[2].trim() });
      }
    }

    if (dsl?.rules?.length) {
      relations.push(...dsl.rules.map((rule) => ({ relation: rule.relation, from: rule.from, to: rule.to })));
    }

    return uniqueBy(relations, (value) => `${value.relation}:${value.from.toLowerCase()}:${value.to.toLowerCase()}`);
  }

  _taskGraph(raw, taskType, entities, constraints, relations, dsl = null) {
    const nodes = [
      { id: "intent", type: "intent", label: taskType, text: raw },
      { id: "query", type: "query", label: raw.slice(0, 120), text: raw },
    ];
    const edges = [
      { from: "intent", to: "query", relation: "describes" },
    ];

    entities.forEach((entity, index) => {
      const id = `entity:${index}`;
      nodes.push({ id, type: "entity", label: entity, text: entity });
      edges.push({ from: "query", to: id, relation: "mentions" });
    });

    constraints.forEach((constraint, index) => {
      const id = `constraint:${index}`;
      nodes.push({ id, type: "constraint", label: constraint, text: constraint });
      edges.push({ from: "intent", to: id, relation: "constrains" });
    });

    relations.forEach((relation, index) => {
      const id = `relation:${index}`;
      nodes.push({ id, type: "relation", label: relation.relation, text: `${relation.from} ${relation.relation} ${relation.to}` });
      edges.push({ from: "query", to: id, relation: relation.relation });
    });

    dsl?.tasks?.forEach((task, index) => {
      const id = `task:${index}`;
      nodes.push({ id, type: "task", label: task, text: task });
      edges.push({ from: "intent", to: id, relation: "task" });
    });

    return { nodes, edges };
  }

  _confidence(taskType, tokens, constraints, relations) {
    let confidence = 0.5;
    if (tokens.length > 4) confidence += 0.1;
    if (constraints.length) confidence += 0.15;
    if (relations.length) confidence += 0.1;
    if (taskType !== "analysis") confidence += 0.1;
    return Math.min(0.95, confidence);
  }
}

export class KnowledgeGraph {
  constructor() {
    this.triples = [];
    this.index = new Map();
    this.nodes = new Set();
  }

  addText(text, source = "corpus") {
    const sentences = normalizeText(text)
      .split(/[.!?]+\s*/)
      .map((sentence) => sentence.trim())
      .filter(Boolean);

    for (const sentence of sentences) {
      for (const triple of this._extractTriples(sentence, source)) {
        this._addTriple(triple);
      }
    }
  }

  _addTriple(triple) {
    const id = `${triple.subject}|${triple.relation}|${triple.object}`.toLowerCase();
    if (this.triples.some((item) => item.id === id)) return;

    const record = { id, ...triple };
    this.triples.push(record);
    this.nodes.add(record.subject);
    this.nodes.add(record.object);

    for (const term of keywords(`${record.subject} ${record.relation} ${record.object}`)) {
      if (!this.index.has(term)) this.index.set(term, new Set());
      this.index.get(term).add(id);
    }
  }

  _extractTriples(sentence, source) {
    const triples = [];
    const patterns = [
      { relation: "is", regex: /^(.+?)\s+is\s+(?:an?|the)?\s*(.+)$/i },
      { relation: "are", regex: /^(.+?)\s+are\s+(?:an?|the)?\s*(.+)$/i },
      { relation: "uses", regex: /^(.+?)\s+uses\s+(.+)$/i },
      { relation: "requires", regex: /^(.+?)\s+requires\s+(.+)$/i },
      { relation: "depends_on", regex: /^(.+?)\s+depends? on\s+(.+)$/i },
      { relation: "enables", regex: /^(.+?)\s+enables\s+(.+)$/i },
      { relation: "prevents", regex: /^(.+?)\s+prevents\s+(.+)$/i },
      { relation: "supports", regex: /^(.+?)\s+supports\s+(.+)$/i },
      { relation: "contains", regex: /^(.+?)\s+contains\s+(.+)$/i },
      { relation: "includes", regex: /^(.+?)\s+includes\s+(.+)$/i },
      { relation: "maps_to", regex: /^(.+?)\s+maps? to\s+(.+)$/i },
    ];

    for (const { relation, regex } of patterns) {
      const match = sentence.match(regex);
      if (!match) continue;

      const subject = match[1].trim();
      const object = match[2].trim();
      if (subject.length < 2 || object.length < 2) continue;

      triples.push({ subject, relation, object, sentence, source });
    }

    return triples;
  }

  query(text, topK = 5) {
    const queryTerms = keywords(text);
    const scored = this.triples.map((triple) => {
      const tripleText = `${triple.subject} ${triple.relation} ${triple.object}`;
      const score = Math.max(overlapScore(text, tripleText), overlapScore(text, triple.sentence)) + 0.15 * queryTerms.filter((term) => tripleText.toLowerCase().includes(term)).length;
      return { ...triple, score };
    });

    return scored
      .sort((a, b) => b.score - a.score || a.subject.localeCompare(b.subject))
      .slice(0, topK)
      .map((triple) => ({
        id: triple.id,
        type: "fact",
        source: triple.source,
        relation: triple.relation,
        subject: triple.subject,
        object: triple.object,
        content: `${triple.subject} ${triple.relation} ${triple.object}`,
        sentence: triple.sentence,
        score: triple.score,
      }));
  }

  factsForEntity(entity, topK = 5) {
    const term = String(entity || "").toLowerCase();
    return this.triples
      .filter((triple) => triple.subject.toLowerCase().includes(term) || triple.object.toLowerCase().includes(term))
      .slice(0, topK)
      .map((triple) => ({
        id: triple.id,
        type: "fact",
        source: triple.source,
        relation: triple.relation,
        subject: triple.subject,
        object: triple.object,
        content: `${triple.subject} ${triple.relation} ${triple.object}`,
        sentence: triple.sentence,
        score: 1,
      }));
  }
}

export class Retriever {
  constructor(memory, graph) {
    this.memory = memory;
    this.graph = graph;
  }

  pullFacts(query, topK = 5) {
    const memoryLong = this.memory.retrieveDetailed(query, topK, "long").map((item, index) => ({
      id: `memory-long:${index}`,
      type: "chunk",
      source: "memory.long",
      content: item.chunk,
      score: item.score,
    }));

    const memoryShort = this.memory.retrieveDetailed(query, Math.max(2, Math.min(topK, 3)), "short").map((item, index) => ({
      id: `memory-short:${index}`,
      type: "chunk",
      source: "memory.short",
      content: item.chunk,
      score: item.score * 0.95,
    }));

    const graphFacts = this.graph.query(query, topK);

    return uniqueBy([...memoryLong, ...memoryShort, ...graphFacts], (item) => item.content.toLowerCase()).slice(0, topK * 2);
  }

  retrieve(query, topK = 5) {
    return this.pullFacts(query, topK);
  }
}

export class ConstraintSolver {
  solve(items, relations = [], options = {}) {
    const nodes = uniqueBy(items.map((item) => normalizeText(item)).filter(Boolean), (value) => value.toLowerCase());
    const index = new Map(nodes.map((node, position) => [node.toLowerCase(), position]));
    const edges = new Map(nodes.map((node) => [node, new Set()]));
    const indegree = new Map(nodes.map((node) => [node, 0]));
    const conflicts = [];
    const predecessorMap = new Map(nodes.map((node) => [node, new Set()]));

    for (const relation of relations) {
      const from = this._resolve(relation.from, index, nodes);
      const to = this._resolve(relation.to, index, nodes);
      if (!from || !to) {
        conflicts.push(`unresolved relation: ${relation.from} ${relation.relation} ${relation.to}`);
        continue;
      }

      if (relation.relation === "before") {
        this._edge(from, to, edges, indegree);
        predecessorMap.get(to)?.add(from);
      } else if (relation.relation === "after") {
        this._edge(to, from, edges, indegree);
        predecessorMap.get(from)?.add(to);
      } else if (relation.relation === "depends_on" || relation.relation === "requires") {
        this._edge(to, from, edges, indegree);
        predecessorMap.get(from)?.add(to);
      } else if (relation.relation === "mutex") {
        conflicts.push(`mutex violation: ${from} ! ${to}`);
      }
    }

    const queue = nodes.filter((node) => indegree.get(node) === 0).sort((a, b) => a.localeCompare(b));
    const ordered = [];

    while (queue.length) {
      const node = queue.shift();
      ordered.push(node);

      for (const next of edges.get(node) || []) {
        indegree.set(next, indegree.get(next) - 1);
        if (indegree.get(next) === 0) queue.push(next);
      }

      queue.sort((a, b) => a.localeCompare(b));
    }

    if (ordered.length !== nodes.length) {
      conflicts.push("constraint cycle detected");
      return { ok: false, ordered, conflicts };
    }

    const schedule = this._schedule(ordered, predecessorMap, options);
    conflicts.push(...schedule.conflicts);

    return { ok: conflicts.length === 0, ordered, conflicts, schedule: schedule.items, makespan: schedule.makespan };
  }

  _resolve(value, index, nodes) {
    const normalized = normalizeText(value);
    if (!normalized) return null;
    if (index.has(normalized.toLowerCase())) return nodes[index.get(normalized.toLowerCase())];
    return nodes.find((node) => node.toLowerCase().includes(normalized.toLowerCase())) || null;
  }

  _edge(from, to, edges, indegree) {
    if (!edges.has(from)) edges.set(from, new Set());
    if (!edges.get(from).has(to)) {
      edges.get(from).add(to);
      indegree.set(to, (indegree.get(to) || 0) + 1);
    }
  }

  _schedule(ordered, predecessorMap, options = {}) {
    const taskSpecs = options.taskSpecs || [];
    const resources = options.resources || {};
    const conflicts = [];
    const specMap = new Map(taskSpecs.map((task) => [task.name.toLowerCase(), task]));
    const allocations = new Map();
    const scheduled = new Map();

    for (const taskName of ordered) {
      const spec = specMap.get(taskName.toLowerCase()) || { name: taskName, duration: 1, resource: null };
      const duration = Math.max(1, Number(spec.duration || 1));
      const resource = spec.resource || null;
      const predecessors = [...(predecessorMap.get(taskName) || [])];
      let start = predecessors.reduce((maxEnd, predecessor) => {
        const scheduledPred = scheduled.get(predecessor);
        return Math.max(maxEnd, scheduledPred ? scheduledPred.end : 0);
      }, 0);

      if (resource) {
        const capacity = Math.max(1, Number(resources[resource] || 1));
        if (!allocations.has(resource)) allocations.set(resource, []);
        start = this._nextAvailableStart(start, duration, allocations.get(resource), capacity);
      }

      const item = {
        task: taskName,
        duration,
        resource,
        start,
        end: start + duration,
      };

      scheduled.set(taskName, item);
      if (resource) allocations.get(resource).push(item);
    }

    const items = ordered.map((taskName) => scheduled.get(taskName)).filter(Boolean);
    const makespan = items.reduce((maxEnd, item) => Math.max(maxEnd, item.end), 0);

    return { items, makespan, conflicts };
  }

  _nextAvailableStart(start, duration, allocations, capacity) {
    let candidate = start;
    while (true) {
      let exceeds = false;
      for (let time = candidate; time < candidate + duration; time++) {
        const concurrent = allocations.filter((item) => item.start <= time && item.end > time).length;
        if (concurrent >= capacity) {
          exceeds = true;
          break;
        }
      }
      if (!exceeds) return candidate;
      candidate += 1;
    }
  }
}

export class PlannerExecutor {
  plan(intent, evidence = []) {
    const steps = [
      { id: "parse", action: "parse_intent", dependsOn: [] },
      { id: "retrieve", action: "retrieve_evidence", dependsOn: ["parse"] },
      { id: "reason", action: "reason_over_evidence", dependsOn: ["retrieve"] },
      { id: "rank", action: "rank_candidates", dependsOn: ["reason"] },
      { id: "verify", action: "verify_answer", dependsOn: ["rank"] },
    ];

    if (intent.taskType === "planner" || intent.structuredQuery.wantsPlanning) {
      steps.splice(2, 0, { id: "solve", action: "solve_constraints", dependsOn: ["retrieve"] });
      steps.push({ id: "execute", action: "execute_plan", dependsOn: ["verify"] });
    }

    return {
      intent: intent.taskType,
      steps,
      evidenceIds: evidence.map((item) => item.id),
    };
  }

  execute(plan, handlers) {
    const outputs = {};
    for (const step of plan.steps) {
      const handler = handlers[step.action];
      if (typeof handler !== "function") continue;
      outputs[step.id] = handler({ plan, outputs });
    }
    return outputs;
  }
}

export class Reasoner {
  constructor(constraintSolver = new ConstraintSolver()) {
    this.constraintSolver = constraintSolver;
  }

  infer(intent, evidence, graph) {
    const evidenceLines = evidence.map((item) => item.content || item.sentence || "").filter(Boolean);
    const facts = graph.query(intent.raw, 5);
    const reasoning = [];

    if (intent.taskType === "planner" && intent.relations.length) {
      const planItems = intent.dsl?.tasks?.length
        ? intent.dsl.tasks
        : intent.entities.length
          ? intent.entities
          : evidenceLines.slice(0, 6);
      const solve = this.constraintSolver.solve(planItems, intent.relations, {
        taskSpecs: intent.dsl?.taskSpecs || [],
        resources: intent.dsl?.resources || {},
      });
      reasoning.push(`constraint solver: ${solve.ok ? "consistent" : "conflict"}`);
      if (solve.ordered.length) reasoning.push(`order: ${solve.ordered.join(" -> ")}`);
      if (solve.schedule?.length) reasoning.push(`makespan: ${solve.makespan}`);
      if (solve.conflicts.length) reasoning.push(...solve.conflicts);
      return { reasoning, conclusion: solve.ordered, facts, solver: solve };
    }

    if (intent.taskType === "comparison" && intent.entities.length >= 2) {
      reasoning.push(`compare ${intent.entities[0]} with ${intent.entities[1]}`);
      return {
        reasoning,
        conclusion: facts.slice(0, 3).map((fact) => fact.content),
        facts,
      };
    }

    if (intent.taskType === "verifier") {
      reasoning.push("verify consistency against retrieved evidence");
      return { reasoning, conclusion: facts.slice(0, 3).map((fact) => fact.content), facts };
    }

    reasoning.push(`use ${evidenceLines.length} evidence items`);
    reasoning.push(`selected facts: ${facts.slice(0, 3).map((fact) => fact.content).join("; ")}`);

    return {
      reasoning,
      conclusion: evidenceLines.slice(0, 3),
      facts,
    };
  }
}

export class Ranker {
  scoreCandidate(candidate, evidence, intent) {
    const answerText = normalizeText(candidate.answer || candidate.text || "");
    const evidenceText = evidence.map((item) => item.content || item.sentence || "").join(" ");
    const queryText = intent.raw || intent.normalized || "";
    const relevance = overlapScore(queryText, answerText) + 0.5 * overlapScore(queryText, evidenceText);
    const coverage = candidate.solverBacked
      ? 1
      : evidence.length
        ? (candidate.evidenceIds ? new Set(candidate.evidenceIds).size / evidence.length : 0)
        : 0;
    const consistency = candidate.verification?.ok ? 1 : candidate.verification?.supportRatio ?? 0.25;
    const structuredBonus = candidate.kind === "evidence" ? 0.15 : candidate.kind === "graph" ? 0.1 : candidate.kind === "plan" ? 0.2 : 0;
    const unsupportedPenalty = candidate.verification?.issues?.length ? Math.min(0.35, candidate.verification.issues.length * 0.08) : 0;
    const plannerBoost = intent.taskType === "planner" && candidate.solverBacked && candidate.solver?.ok ? 0.4 : 0;

    return 0.45 * relevance + 0.3 * coverage + 0.2 * consistency + structuredBonus + plannerBoost - unsupportedPenalty;
  }

  rank(candidates, evidence, intent) {
    return candidates
      .map((candidate) => ({ ...candidate, score: this.scoreCandidate(candidate, evidence, intent) }))
      .sort((a, b) => b.score - a.score || a.kind.localeCompare(b.kind));
  }
}

export class Verifier {
  verify(candidate, evidence, intent) {
    const issues = [];
    const answerText = normalizeText(candidate.answer || candidate.text || "");
    const evidenceText = evidence.map((item) => item.content || item.sentence || "").join(" ");
    const evidenceTerms = new Set(keywords(evidenceText));
    const answerTerms = keywords(answerText);
    const supported = answerTerms.filter((term) => evidenceTerms.has(term));
    const supportRatio = answerTerms.length ? supported.length / answerTerms.length : 0;

    if (!answerText) issues.push("empty answer");
    if (!evidence.length && !candidate.solverBacked) issues.push("no evidence retrieved");
    if (supportRatio < 0.2 && evidence.length && !candidate.solverBacked) issues.push("insufficient evidence linkage");

    if (candidate.evidenceIds && candidate.evidenceIds.length === 0) issues.push("candidate lacks citations");
    if (candidate.kind === "plan" && intent.taskType !== "planner") issues.push("plan candidate used for non-planning intent");
    if (candidate.solverBacked && candidate.solver?.conflicts?.length) issues.push(...candidate.solver.conflicts);

    if (/\b(?:kill|harm|explosive|weapon|self-harm)\b/i.test(answerText)) {
      issues.push("unsafe content");
    }

    return {
      ok: issues.length === 0,
      issues,
      supportRatio,
      supportedTerms: supported,
    };
  }
}

export class HybridSymbolicSystem {
  constructor(opts = {}) {
    this.order = opts.order ?? 9;
    this.temperature = opts.temperature ?? 0.9;
    this.maxLength = opts.maxLength ?? 400;
    this.topK = opts.topK ?? 3;

    this.parser = new IntentParser();
    this.memory = opts.memory;
    this.graph = opts.graph ?? new KnowledgeGraph();
    this.retriever = new Retriever(this.memory, this.graph);
    this.reasoner = new Reasoner();
    this.planner = new PlannerExecutor();
    this.ranker = new Ranker();
    this.verifier = new Verifier();
    this.markov = opts.markov;
  }

  learn(corpus) {
    const text = normalizeText(corpus);
    if (!text) {
      return { chars: 0, facts: 0, chunks: 0 };
    }

    this.memory.addText(text);
    this.graph.addText(text);
    this.markov.fit(text);

    return {
      chars: text.length,
      chunks: this.memory.chunkCount,
      facts: this.graph.triples.length,
      transitions: this.markov.transitionCount,
      keywords: this.memory.topKeywords(8),
    };
  }

  _buildEvidenceDigest(intent, evidence, reasoning, topK = this.topK) {
    const selected = evidence.slice(0, topK);
    const lines = selected.map((item, index) => `[E${index + 1}] ${item.content || item.sentence || ""}`);
    const facts = reasoning.facts?.slice(0, topK).map((fact, index) => `[F${index + 1}] ${fact.content}`) || [];
    const body = [...lines, ...facts];

    return {
      kind: "evidence",
      answer: body.length
        ? [`Answer:`, ...body, reasoning.reasoning.length ? `Reasoning: ${reasoning.reasoning.join(" | ")}` : "", `Query: ${intent.raw}`]
            .filter(Boolean)
            .join("\n")
        : `Answer: No sufficiently strong evidence was retrieved for: ${intent.raw}`,
      evidenceIds: selected.map((item) => item.id),
      details: body,
    };
  }

  _buildGraphCandidate(intent, evidence, reasoning) {
    if (!reasoning.conclusion || !reasoning.conclusion.length) return null;
    return {
      kind: "graph",
      answer: [`Answer:`, ...reasoning.conclusion.map((item, index) => `[G${index + 1}] ${item}`), `Query: ${intent.raw}`].join("\n"),
      evidenceIds: evidence.slice(0, 3).map((item) => item.id),
    };
  }

  _buildPlanCandidate(intent, reasoning) {
    if (intent.taskType !== "planner" || !reasoning.solver) return null;

    const ordered = reasoning.solver.ordered.map((task, index) => `[P${index + 1}] ${task}`);
    const status = reasoning.solver.ok ? "consistent" : `conflict: ${reasoning.solver.conflicts.join("; ")}`;

    return {
      kind: "plan",
      answer: [
        "Answer:",
        `Plan status: ${status}`,
        ...ordered,
        ...(reasoning.solver.schedule?.length
          ? reasoning.solver.schedule.map((item) => `[S] ${item.task}: t=${item.start}->${item.end}${item.resource ? ` [${item.resource}]` : ""}`)
          : []),
        ...(Object.keys(intent.dsl?.resources || {}).length
          ? [`Resources: ${Object.entries(intent.dsl.resources).map(([name, cap]) => `${name}=${cap}`).join(", ")}`]
          : []),
        `Makespan: ${reasoning.solver.makespan ?? 0}`,
        `Rules: ${(intent.dsl?.rules || intent.relations).map((rule) => `${rule.from} ${rule.relation} ${rule.to}`).join(" | ")}`,
        `Query: ${intent.raw}`,
      ].join("\n"),
      evidenceIds: ["solver"],
      solverBacked: true,
      solver: reasoning.solver,
    };
  }

  _buildMarkovCandidate(intent, evidence, maxLength) {
    const context = evidence.map((item) => item.content || item.sentence || "").join(" ").slice(0, 600);
    const prompt = `Evidence: ${context} Question: ${intent.raw} Answer:`;
    return {
      kind: "generation",
      answer: this.markov.generateBest(prompt, maxLength, Math.max(4, this.topK + 4)),
      evidenceIds: evidence.slice(0, this.topK).map((item) => item.id),
      prompt,
    };
  }

  think(query, opts = {}) {
    const parsed = this.parser.parse(query);
    const order = opts.order ?? this.order;
    const temperature = opts.temperature ?? this.temperature;
    const maxLength = opts.maxLength ?? this.maxLength;
    const topK = opts.topK ?? this.topK;

    this.markov.order = order;
    this.markov.temperature = temperature;

    const plan = this.planner.plan(parsed, []);
    const evidence = this.retriever.pullFacts(parsed.raw, topK * 2);
    const reasoning = this.reasoner.infer(parsed, evidence, this.graph);

    const candidates = [
      this._buildPlanCandidate(parsed, reasoning),
      this._buildEvidenceDigest(parsed, evidence, reasoning, topK),
      this._buildGraphCandidate(parsed, evidence, reasoning),
      this._buildMarkovCandidate(parsed, evidence, maxLength),
    ].filter(Boolean);

    const evaluated = this.ranker.rank(
      candidates.map((candidate) => ({ ...candidate, verification: this.verifier.verify(candidate, evidence, parsed) })),
      evidence,
      parsed,
    );

    const accepted = evaluated.find((candidate) => candidate.verification.ok) || evaluated[0];
    const verification = this.verifier.verify(accepted, evidence, parsed);

    const answer = accepted.solverBacked ? accepted.answer : accepted.answer.slice(0, maxLength);

    return {
      query: parsed.raw,
      answer,
      intent: parsed,
      plan,
      evidence,
      reasoning,
      rankedCandidates: evaluated,
      verification,
      retrieved: evidence.map((item) => item.content || item.sentence || ""),
      prompt: accepted.prompt || `Evidence-linked response for: ${parsed.raw}`,
      stats: {
        order,
        temperature,
        answerLength: answer.length,
        transitions: this.markov.transitionCount,
        chunks: this.memory.chunkCount,
        facts: this.graph.triples.length,
        evidenceCount: evidence.length,
        verificationOk: verification.ok,
        taskType: parsed.taskType,
      },
    };
  }
}