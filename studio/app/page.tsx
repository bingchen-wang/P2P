"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Boxes,
  ChevronRight,
  CircleDollarSign,
  CircleHelp,
  Database,
  FileCode2,
  FileImage,
  FileText,
  FlaskConical,
  FolderOpen,
  Gauge,
  History,
  Home as HomeIcon,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  Square,
  TerminalSquare,
  TriangleAlert,
} from "lucide-react";
import { toast, Toaster } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ResultsView } from "@/app/results";
import { populationFamily, populationFamilyColors } from "@/app/population-style";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type View = "home" | "new" | "jobs" | "runs" | "budget" | "data" | "backends";
type Interval = { low?: number; point?: number; high?: number };
type Metric = { metric: string; value: number; split?: string; unit?: string; predictionPolicy?: string; qidsRequested?: number; qidsUsed?: number };
type PopulationSummary = {
  generated: number;
  selected: number;
  antiTypes: number;
  sourcePath: string;
  modes: { mode: string; family?: string; generated: number; selected: number; antiTypes: number }[];
};
type Manifest = {
  path: string;
  name: string;
  seed?: number;
  mode?: string;
  targetN?: number;
  method?: string;
  paid: boolean;
  backend: { module: string; type: string; model: string }[];
};
type Run = {
  path: string;
  runId: string;
  repeat?: number;
  seed?: number;
  conditions: Record<string, unknown>;
  method?: string;
  solverStatus?: string;
  metrics: Metric[];
  usage: Record<string, number>;
  questionCount: number;
  hasDiagnostics: boolean;
  population?: PopulationSummary | null;
  modifiedAt: string;
  origin: "current" | "legacy";
  provenance?: { label?: string; sourceRepository?: string; sourcePath?: string; cleaningContract?: string };
};
type Report = {
  path: string;
  name: string;
  title: string;
  kind: "pipeline" | "regression" | "generation" | "elicitation";
  models: string[];
  metrics: {
    model?: string;
    testMse?: number;
    testOneMinusJsd?: number;
    testEmd?: number;
    testAccuracy?: number;
    selected?: number;
  }[];
  reportCount: number;
  imageCount: number;
  totalCost?: number;
  numEndowments?: number;
  modifiedAt: string;
};
type Budget = {
  path: string;
  confidence?: number;
  historicalRuns?: number;
  pilotRecords?: number;
  configSource?: string;
  modifiedAt: string;
  scenarios: {
    name: string;
    model?: string;
    tokens: Interval;
    cost: Interval;
    status?: string;
    budgetUsd?: number;
  }[];
};
type Backend = {
  path: string;
  name: string;
  type: string;
  local: boolean;
  credentialName?: string;
  credentialAvailable: boolean;
  modules: { module: string; type: string; model: string }[];
};
type BackendCapability = {
  id: string;
  local: boolean;
  metered: boolean;
  dependency?: string;
  dependencyAvailable: boolean;
  credentialName?: string;
  credentialAvailable: boolean;
  modelRequired?: boolean;
  options?: { name: string; default: unknown; required: boolean }[];
};
type Survey = {
  path: string;
  name: string;
  questionId?: string;
  questionText?: string;
  optionMapping?: string;
  modifiedAt: string;
};
type Job = {
  id: string;
  command: string;
  config: string;
  output: string;
  status: "running" | "paused" | "stopped" | "failed" | "complete" | "interrupted";
  exitCode?: number;
  pid?: number;
  createdAt: string;
  log: string[];
  recovery: "live" | "inspect" | "checkpoint" | "new-output";
  solver?: string | null;
  durable: boolean;
};
type Workspace = {
  counts: { runs: number; currentRuns: number; historicalRuns: number; reports: number; aggregates?: number; manifests: number; datasets: number };
  privacy: { boundary: string; network: string; providers: string };
};
type MetricInfo = { id: string; label: string; short: string; direction: "lower" | "higher"; unit: string; description: string };
type ChoiceInfo = { id: string; label: string; description: string };
type Stage2Configuration = { id: string; label: string; description: string; options: Record<string, unknown> };
type Capabilities = {
  workflows: { id: GuidedCommand; label: string; needsGeneration: boolean; needsModel: boolean; needsTarget: boolean }[];
  generationModes: { id: "active" | "vanilla"; label: string; advanced: boolean }[];
  models: { id: string; label: string; guided: boolean; controls: string[]; description?: string; method?: string; options?: Record<string, unknown>; zeroThreshold?: number }[];
  backends: BackendCapability[];
  metrics: MetricInfo[];
  stage2: {
    configurations: Stage2Configuration[];
    defaultConfiguration: string;
    validationStrategies: string[];
    foldMethods: string[];
    defaultFoldMethod: string;
    alphaGrid: { logspace: [number, number, number] };
    l1RatioGrid: { linspace: [number, number, number] };
    solvers: string[];
    defaultSolver: string;
    solverFallbacks: string[];
    solverCommands: string[];
    zeroThreshold: number;
    reference: { conflicts: string[]; excludedMethods: string[] };
  };
  evaluation: { distributionPolicies: ChoiceInfo[]; invalidDistribution: ChoiceInfo[] };
  defaults: { seed: number; generationMode: "active" | "vanilla"; targetN: number; modelMethod?: string; cvFolds: number; holdoutSplit: string; metrics: string[]; distributionPredictionPolicy: string; invalidDistribution: string };
  processControls: { pause: boolean; resume: boolean; stop: boolean; platform: string };
};
type Dataset = {
  id: string; name: string; path: string; source: string; schema: string; target?: string;
  questionCount?: number; splitCounts: Record<string, number>; digests: Record<string, string>; portable: boolean;
};
type Bootstrap = {
  repository: string;
  root: string;
  localOnly: boolean;
  workspace: Workspace;
  capabilities: Capabilities;
  manifests: Manifest[];
  runs: Run[];
  reports: Report[];
  budgets: Budget[];
  surveys: Survey[];
  backends: Backend[];
  datasets: Dataset[];
  jobs: Job[];
};
type GuidedCommand = "pipeline" | "generate" | "regress" | "evaluate";
type ExperimentDraft = {
  command: GuidedCommand;
  name: string;
  seed: number;
  surveySource: string;
  surveySchema: string;
  target: string;
  // A grouped target file's subgroup; empty for a flat file or its default.
  targetSubgroup: string;
  // Optional reference distribution: Stage 2 then fits each target's change from it.
  referenceDistribution: string;
  basis: string;
  predictions: string;
  generationMode: "vanilla" | "active";
  targetN: number;
  initialN: number;
  updateSteps: number;
  attributeBank: string;
  backendProfile: string;
  modelMethod: string;
  cvFolds: number;
  configuration: string;
  validationStrategy: "cv" | "holdout";
  holdoutSplit: string;
  foldMethod: string;
  foldFile: string;
  blockWeighting: boolean;
  // Each question's prediction sums exactly to one instead of the soft mass penalty.
  exactQuestionMass: boolean;
  // Empty means the method's default.
  zeroThreshold: string;
  solver: string;
  alphaGrid: [number, number, number];
  l1RatioGrid: [number, number, number];
  metrics: string[];
  distributionPolicy: string;
  invalidDistribution: string;
};
type CreatedManifest = { path: string; content: string; manifest: Manifest };
type RunDetail = {
  model?: { method?: string; solver?: string; status?: string; coefficients?: Record<string, number> };
  questions?: {
    qid: string;
    question: string;
    split: string;
    option_order: string[];
    target: Record<string, number>;
    prediction: Record<string, number>;
  }[];
  metrics?: { metrics?: Metric[] };
  usage?: { summary?: Record<string, number>; records?: Record<string, unknown>[] };
};
type PredictionQuestion = NonNullable<RunDetail["questions"]>[number];
type ReportDetail = {
  summary: Report;
  documents: { name: string; title: string; content: string }[];
  images: { name: string; path: string; category: "predictions" | "diagnostics" | "generation" | "weights" | "other"; size: number }[];
  metadata: Record<string, unknown>;
  results: Record<string, unknown>;
  tokenUsage: { total_cost?: number; per_model_module?: Record<string, Record<string, number>> };
  metricRows: Record<string, string>[];
  modeRows: Record<string, string>[];
  entropyRows: Record<string, string>[];
  aggregateEntropy: Record<string, string>[];
  files: { name: string; size: number; type: string }[];
};
type BudgetDetail = {
  notes?: string[];
  scenarios?: {
    name: string;
    budget_status?: string;
    budget_usd?: number;
    total_tokens: Interval;
    total_cost_usd: Interval;
    modules?: {
      module_name: string;
      model_name: string;
      coverage?: string;
      input_tokens: Interval;
      output_tokens: Interval;
      cost_usd: Interval;
    }[];
  }[];
};
type DatasetInspection = {
  valid: boolean;
  source: string;
  schema?: string;
  target?: string;
  headers: string[];
  mapping: Record<string, string | null>;
  questionCount: number;
  optionRowCount: number;
  splitCounts: Record<string, number>;
  targetCoverage: { matched: number; missing: number; extra: number };
  preview: { qid: string; question: string; split: string; options: Record<string, string> }[];
  errors: string[];
  warnings: string[];
};
type ManifestPreview = {
  valid: boolean;
  path: string;
  output?: string;
  content: string;
  configDigest: string;
  paid: boolean;
  costStatus: "unknown" | "offline-or-local";
  warnings: string[];
  solver?: string | null;
  cli: string;
  targetSubgroups?: { subgroups: string[]; default: string | null } | null;
};
type BackendCheck = {
  path: string;
  name: string;
  type: string;
  ready: boolean;
  checks: { label: string; ok: boolean; detail: string }[];
  networkCallMade: boolean;
};
type ManifestDetail = { path: string; command: GuidedCommand; config: Record<string, any>; summary: Manifest };

const navItems: { id: View; label: string; icon: typeof FlaskConical }[] = [
  { id: "home", label: "Home", icon: HomeIcon },
  { id: "new", label: "Experiments", icon: FlaskConical },
  { id: "runs", label: "Runs", icon: History },
  { id: "budget", label: "Budgets", icon: CircleDollarSign },
  { id: "data", label: "Data sources", icon: Database },
  { id: "backends", label: "Backends", icon: Boxes },
];

const welcomePreferenceKey = "p2p.studio.welcome.v1";
const draftPreferenceKey = "p2p.studio.experiment-draft.v1";
const validViews = new Set<View>(navItems.map((item) => item.id));

function readRoute() {
  if (typeof window === "undefined") return { view: "home" as View, run: null as string | null, report: null as string | null, manifest: null as string | null, compare: [] as string[] };
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("view") as View | null;
  // Jobs and the library are one Runs view; older links to ?view=jobs land there.
  return {
    view: requested === "jobs" ? "runs" : requested && validViews.has(requested) ? requested : "home",
    run: params.get("run"),
    report: params.get("report"),
    manifest: params.get("from"),
    compare: params.getAll("compare"),
  };
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body as T;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatNumber(value?: number, digits = 0) {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function formatMoney(value?: number) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}

function runScore(run: Run) {
  return run.metrics.find((metric) => metric.metric === "one_minus_jsd")?.value;
}

function friendlyRunName(value: string) {
  const cleaned = value
    .replace(/[-_]r\d{3,}$/i, "")
    .replace(/stage[_-]?2/gi, "Stage 2")
    .replace(/[_-]+/g, " ")
    .replace(/\bW(\d+)\b/g, "Wave $1")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^Stage 2 ablation\s+/i, "Stage 2 ablation · ")
    .replace(/\s+(Wave \d+)$/i, " · $1");
  return cleaned ? cleaned.split(" · ").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" · ") : value;
}

function PopulationMiniature({ population }: { population?: PopulationSummary | null }) {
  if (!population?.generated || !population.modes.length) return null;
  const maximumVisible = 400;
  const cells = population.modes.flatMap((mode) => {
    const visible = population.generated <= maximumVisible
      ? mode.generated
      : Math.max(1, Math.round(mode.generated / population.generated * maximumVisible));
    const selected = Math.min(visible, Math.round(mode.selected / Math.max(1, mode.generated) * visible));
    const antiTypes = Math.min(selected, Math.round(mode.antiTypes / Math.max(1, mode.generated) * visible));
    const family = populationFamily(mode.mode, mode.family);
    return Array.from({ length: visible }, (_, index) => ({
      family,
      selected: index < selected,
      antiType: index < antiTypes,
      mode: mode.mode,
    }));
  }).slice(0, maximumVisible);
  const columns = Math.max(8, Math.ceil(Math.sqrt(cells.length * 1.25)));
  const families = [...new Set(cells.map((cell) => cell.family))];
  return <div className="population-miniature"><div className="population-miniature-head"><span><small>Stage 1 population → Stage 2 ensemble</small><strong>{formatNumber(population.generated)} generated <i>→</i> {formatNumber(population.selected)} selected</strong></span>{population.antiTypes > 0 && <span className="population-anti-count">{formatNumber(population.antiTypes)} anti-{population.antiTypes === 1 ? "type" : "types"}</span>}</div><div className="population-miniature-grid" style={{ "--population-cols": columns } as CSSProperties} aria-label={`${population.generated} generated proxies, ${population.selected} selected, ${population.antiTypes} anti-types`}>{cells.map((cell, index) => <i key={`${cell.mode}-${index}`} title={`${cell.mode}${cell.selected ? " · selected" : " · not selected"}${cell.antiType ? " · anti-type" : ""}`} className={`population-mini-agent${cell.selected ? " population-mini-agent-selected" : ""}${cell.antiType ? " population-mini-agent-anti" : ""}`} style={{ "--population-color": populationFamilyColors[cell.family] } as CSSProperties} />)}</div><div className="population-miniature-foot"><span>{cells.length < population.generated ? `${cells.length}-proxy proportional view` : "Complete endowment budget"}</span><span className="population-family-key">{families.map((family) => <span key={family}><i style={{ backgroundColor: populationFamilyColors[family] }} />{family}</span>)}</span></div></div>;
}

function HeldOutPredictionPreview({ questions }: { questions: PredictionQuestion[] | null }) {
  const rawPoints = (questions || []).flatMap((question) => {
    const options = question.option_order?.length
      ? question.option_order
      : [...new Set([...Object.keys(question.target || {}), ...Object.keys(question.prediction || {})])];
    return options.flatMap((option) => {
      const target = Number(question.target?.[option]);
      const prediction = Number(question.prediction?.[option]);
      return Number.isFinite(target) && Number.isFinite(prediction) ? [{ target, prediction }] : [];
    });
  });
  const divisor = Math.max(0, ...rawPoints.flatMap((point) => [point.target, point.prediction])) > 1.5 ? 100 : 1;
  const allPoints = rawPoints.map((point) => ({
    target: Math.max(0, Math.min(1, point.target / divisor)),
    prediction: Math.max(0, Math.min(1, point.prediction / divisor)),
  }));
  const points = allPoints.length <= 220
    ? allPoints
    : allPoints.filter((_, index) => index % Math.ceil(allPoints.length / 220) === 0);
  const mae = allPoints.length
    ? allPoints.reduce((sum, point) => sum + Math.abs(point.prediction - point.target), 0) / allPoints.length
    : null;
  const left = 34;
  const top = 18;
  const size = 208;
  const x = (value: number) => left + value * size;
  const y = (value: number) => top + (1 - value) * size;

  return (
    <figure className="home-prediction-preview" aria-label="Held-out target and predicted answer shares">
      <div className="home-prediction-head">
        <span><small>Held-out calibration</small><strong>Target ↔ prediction</strong></span>
        {!!questions?.length && <em>{questions.length} questions</em>}
      </div>
      {questions === null ? (
        <div className="home-prediction-loading" aria-label="Loading held-out predictions"><span /><span /><span /><span /></div>
      ) : allPoints.length ? (
        <svg className="home-prediction-chart" viewBox="0 0 270 264" role="img" aria-label={`${allPoints.length} held-out option shares; mean absolute error ${mae == null ? "unavailable" : `${(mae * 100).toFixed(1)} percentage points`}`}>
          {[0, .25, .5, .75, 1].map((tick) => <g key={tick}><line className="prediction-grid-line" x1={x(tick)} x2={x(tick)} y1={top} y2={top + size} /><line className="prediction-grid-line" x1={left} x2={left + size} y1={y(tick)} y2={y(tick)} /></g>)}
          <line className="prediction-ideal-line" x1={left} y1={top + size} x2={left + size} y2={top} />
          {points.map((point, index) => <circle key={index} className="prediction-share-point" cx={x(point.target)} cy={y(point.prediction)} r="2.35" />)}
          <text className="prediction-axis-label" x={left + size / 2} y="259" textAnchor="middle">Observed share</text>
          <text className="prediction-axis-label" x="9" y={top + size / 2} textAnchor="middle" transform={`rotate(-90 9 ${top + size / 2})`}>Predicted share</text>
          <text className="prediction-tick-label" x={left} y="243" textAnchor="middle">0</text>
          <text className="prediction-tick-label" x={left + size} y="243" textAnchor="middle">1</text>
          <text className="prediction-tick-label" x="26" y={top + size + 3} textAnchor="end">0</text>
          <text className="prediction-tick-label" x="26" y={top + 3} textAnchor="end">1</text>
        </svg>
      ) : (
        <div className="home-prediction-empty"><BarChart3 /><strong>No prediction artifact</strong><p>This record can still be inspected, but it does not expose question-level held-out predictions.</p></div>
      )}
      <figcaption><span><i /> Held-out option shares</span><strong>{mae == null ? "MAE —" : `MAE ${(mae * 100).toFixed(1)} pp`}</strong></figcaption>
    </figure>
  );
}

function PopulationBudgetPreview({ targetN }: { targetN: number }) {
  const total = Math.max(0, Math.round(targetN));
  if (!total) return null;
  const visible = Math.min(total, 180);
  const columns = Math.max(10, Math.ceil(Math.sqrt(visible * 1.4)));
  return <div className="population-budget-preview sm:col-span-2"><div><small>Planned endowment budget</small><strong>{formatNumber(total)} proxies</strong><p>The formation reflects the requested Stage 1 population size. Mode colours are assigned only after the attribute bank actually generates them.</p></div><div className="population-budget-grid" style={{ "--population-cols": columns } as CSSProperties} aria-label={`Planned population of ${total} proxies`}>{Array.from({ length: visible }, (_, index) => <i key={index} />)}</div><div className="population-budget-key"><span>Post-generation family key</span>{Object.entries(populationFamilyColors).map(([family, color]) => <span key={family}><i style={{ backgroundColor: color }} />{family}</span>)}</div></div>;
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function yamlScalar(value: unknown) {
  if (typeof value === "string") return JSON.stringify(value);
  if (value == null) return "null";
  return String(value);
}

function yamlLines(value: unknown, depth = 0): string[] {
  const indent = "  ".repeat(depth);
  if (Array.isArray(value)) return value.flatMap((item) => typeof item === "object" && item != null ? [`${indent}-`, ...yamlLines(item, depth + 1)] : [`${indent}- ${yamlScalar(item)}`]);
  if (typeof value !== "object" || value == null) return [`${indent}${yamlScalar(value)}`];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => {
    if (typeof child !== "object" || child == null) return [`${indent}${key}: ${yamlScalar(child)}`];
    if (Array.isArray(child) && !child.length) return [`${indent}${key}: []`];
    return [`${indent}${key}:`, ...yamlLines(child, depth + 1)];
  });
}

const SIMPLE_AVERAGE = "simple_average";

function guidedAlias(draft: ExperimentDraft, caps: Capabilities) {
  // A guided method such as ridge is a registered method with fixed options.
  return caps.models.find((model) => model.id === draft.modelMethod && model.method);
}

// The Stage-2 configurations that can be combined with a reference distribution.
function referenceConfigurations(caps: Capabilities) {
  return caps.stage2.configurations.filter((item) => !caps.stage2.reference.conflicts.some((name) => Boolean(item.options[name])));
}

function stage2Model(draft: ExperimentDraft, caps: Capabilities): Record<string, unknown> {
  // The simple average weights every agent equally: P2P refuses penalty,
  // validation and configuration options for it, so none are written.
  const alias = guidedAlias(draft, caps);
  const model: Record<string, unknown> = { method: alias?.method || draft.modelMethod };
  if (draft.modelMethod === SIMPLE_AVERAGE) return model;
  model.alpha = { logspace: draft.alphaGrid };
  if (alias) Object.assign(model, alias.options || {});
  else if (draft.modelMethod === "elasticnet") model.l1_ratio = { linspace: draft.l1RatioGrid };
  const configuration = caps.stage2.configurations.find((item) => item.id === draft.configuration);
  Object.assign(model, configuration?.options || {});
  if (draft.exactQuestionMass) model.soft_group_sum = false;
  const zeroThreshold = draft.zeroThreshold.trim() ? Number(draft.zeroThreshold) : alias?.zeroThreshold;
  if (zeroThreshold != null && zeroThreshold !== caps.stage2.zeroThreshold) model.zero_threshold = zeroThreshold;
  if (draft.blockWeighting) model.block_weighting = true;
  if (draft.validationStrategy === "holdout") {
    model.validation = { strategy: "holdout", holdout_split: draft.holdoutSplit.trim() || "valid" };
  } else {
    const validation: Record<string, unknown> = { strategy: "cv", cv_folds: draft.cvFolds };
    if (draft.foldMethod === "replay") validation.folds = { method: "replay", file: draft.foldFile.trim() };
    else if (draft.foldMethod !== caps.stage2.defaultFoldMethod) validation.folds = { method: draft.foldMethod };
    model.validation = validation;
  }
  return model;
}

function configurationOf(model: Record<string, any>, caps: Capabilities) {
  const chosen = {
    group_penalty_strength: Number(model.group_penalty_strength ?? 100),
    reference_respondent: Boolean(model.reference_respondent),
    intercept: Boolean(model.intercept),
  };
  const match = caps.stage2.configurations.find((item) => {
    const expected = { group_penalty_strength: 100, reference_respondent: false, intercept: false, ...item.options } as Record<string, unknown>;
    return Number(expected.group_penalty_strength) === chosen.group_penalty_strength
      && Boolean(expected.reference_respondent) === chosen.reference_respondent
      && Boolean(expected.intercept) === chosen.intercept;
  });
  return match?.id || caps.stage2.defaultConfiguration;
}

// The guided method a saved model block expresses: a registered method, or an
// alias such as ridge whose fixed options the block carries.
function guidedMethodOf(model: Record<string, any>, caps: Capabilities): string {
  const alias = caps.models.find((item) => item.method && item.method === model.method && Object.entries(item.options || {}).every(([key, value]) => JSON.stringify(model[key]) === JSON.stringify(value) || (Array.isArray(value) && Array.isArray(model[key]) && value.length === model[key].length && value.every((entry, index) => Number(entry) === Number(model[key][index])))));
  return alias?.id || String(model.method || "");
}

function gridSpec(value: unknown, kind: "logspace" | "linspace", fallback: [number, number, number]): [number, number, number] {
  if (value && typeof value === "object" && !Array.isArray(value) && Array.isArray((value as Record<string, unknown>)[kind])) {
    const spec = (value as Record<string, number[]>)[kind];
    if (spec.length === 3) return [Number(spec[0]), Number(spec[1]), Number(spec[2])];
  }
  // A resolved manifest stores the expanded values; read their range back.
  if (Array.isArray(value) && value.length && value.every((item) => typeof item === "number" && item > 0)) {
    const numbers = value as number[];
    const low = Math.min(...numbers);
    const high = Math.max(...numbers);
    return kind === "logspace"
      ? [Number(Math.log10(low).toFixed(4)), Number(Math.log10(high).toFixed(4)), numbers.length]
      : [Number(low.toFixed(4)), Number(high.toFixed(4)), numbers.length];
  }
  return fallback;
}

function buildGuidedConfig(draft: ExperimentDraft, caps: Capabilities): Record<string, unknown> {
  const config: Record<string, unknown> = { run: { name: draft.name.trim(), seed: draft.seed } };
  const data: Record<string, unknown> = {
    survey: { source: draft.surveySource.trim(), schema: draft.surveySchema.trim() },
  };
  if (["pipeline", "regress", "evaluate"].includes(draft.command)) {
    data.target = draft.targetSubgroup.trim() ? { path: draft.target.trim(), subgroup: draft.targetSubgroup.trim() } : draft.target.trim();
  }
  if (["pipeline", "regress"].includes(draft.command) && draft.referenceDistribution.trim()) data.reference_distribution = draft.referenceDistribution.trim();
  if (draft.command === "regress") data.basis = draft.basis.trim();
  if (draft.command === "evaluate") data.predictions = draft.predictions.trim();
  config.data = data;
  if (["pipeline", "generate"].includes(draft.command)) {
    const generation: Record<string, unknown> = {
      mode: draft.generationMode,
      target_n: draft.targetN,
      batch_size: Math.min(10, draft.targetN),
      delay: 0,
      retry_failed: true,
      max_retries: 1,
      retry_base_delay: 1,
    };
    if (draft.generationMode === "active") {
      generation.initial_n = Math.min(draft.initialN, draft.targetN);
      generation.num_update_steps = draft.updateSteps;
      generation.attribute_bank = { source: draft.attributeBank.trim() };
    }
    config.generation = generation;
    config.backend = { profile: draft.backendProfile.trim() };
  }
  if (["pipeline", "regress"].includes(draft.command)) {
    config.model = stage2Model(draft, caps);
  }
  if (["pipeline", "regress", "evaluate"].includes(draft.command)) {
    config.split = { train_val: ["train", "valid"], test: "test", use_proxy_only: true };
    config.metrics = caps.metrics.map((metric) => metric.id).filter((id) => draft.metrics.includes(id));
    config.evaluation = {
      mse_prediction_policy: "raw",
      distribution_prediction_policy: draft.distributionPolicy,
      invalid_distribution: draft.invalidDistribution,
    };
  }
  return config;
}

function EmptyState({ icon: Icon, title, copy }: { icon: typeof History; title: string; copy: string }) {
  return (
    <div className="surface-card flex min-h-64 flex-col items-center justify-center px-6 py-12 text-center">
      <span className="mb-4 flex size-12 items-center justify-center rounded-2xl bg-primary/8 text-primary"><Icon className="size-5" /></span>
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-1 max-w-md text-sm leading-6 text-muted-foreground">{copy}</p>
    </div>
  );
}

function WorkspaceLoading() {
  return <div className="workspace-loading" aria-live="polite"><div className="workspace-loading-heading"><span /><strong /></div><div className="workspace-loading-inventory">{[0,1,2,3].map((item) => <span key={item} />)}</div><div className="workspace-loading-body"><span /><span /><span /></div><p>Reading the local research record…</p></div>;
}

function MetricBars({ metrics }: { metrics: Metric[] }) {
  if (!metrics.length) return <p className="text-sm text-muted-foreground">No metrics.json artifact found.</p>;
  return (
    <div className="space-y-4">
      {metrics.map((metric) => {
        const width = Math.max(2, Math.min(100, metric.value * 100));
        return (
          <div key={`${metric.metric}-${metric.split}`}>
            <div className="mb-1.5 flex items-center justify-between gap-3 text-sm">
              <span className="font-medium">{metric.metric.replaceAll("_", " ")}</span>
              <span className="font-mono font-semibold">{formatNumber(metric.value, 4)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="viz-metric-fill h-full rounded-full" style={{ width: `${width}%` }} /></div>
          </div>
        );
      })}
    </div>
  );
}

function AppHeader({
  view,
  onNavigate,
  onDocs,
  onRefresh,
  refreshing,
}: {
  view: View;
  onNavigate: (view: View) => void;
  onDocs: () => void;
  onRefresh: () => void;
  refreshing: boolean;
}) {
  return (
    <header className="brand-header sticky top-0 z-30 border-b">
      <div className="brand-topbar h-16 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <span className="brand-mark-shell"><img className="brand-mark shrink-0" src="/p2p-logo-mark-dark.png" alt="P2P chameleon logo" /></span>
          <div className="min-w-0"><p className="brand-name truncate text-[15px] font-semibold"><strong>P2P</strong> Studio</p><p className="text-xs text-muted-foreground">{window.location.hostname === "127.0.0.1" ? "Local research workspace" : "Research workspace"}</p></div>
        </div>
        <nav className="studio-dock hidden lg:flex" aria-label="Research workspace">
          {navItems.map((item) => { const Icon = item.icon; const active = view === item.id; const label = item.id === "runs" ? "Runs" : item.id === "new" ? "New experiment" : item.id === "data" ? "Data" : item.label; return <button key={item.id} type="button" aria-label={label} aria-current={active ? "page" : undefined} data-label={label} className={`studio-dock-item ${active ? "studio-dock-item-active" : ""}`} onClick={() => onNavigate(item.id)}><span className="studio-dock-icon"><Icon /></span></button>; })}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <Badge variant="outline" className="brand-local-badge hidden gap-1.5 font-normal md:inline-flex" title="This Studio reads and writes only this local workspace"><ShieldCheck className="size-3" /><span className="hidden xl:inline">Local</span></Badge>
          <Button variant="ghost" size="sm" className="hidden text-muted-foreground sm:flex" onClick={onDocs} aria-label="Help"><CircleHelp /><span className="hidden xl:inline">Help</span></Button>
          <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={onRefresh} aria-label="Refresh workspace"><RefreshCw className={refreshing ? "animate-spin" : ""} /><span className="hidden sm:inline">Refresh</span></Button>
        </div>
      </div>
    </header>
  );
}

function PageHeading({ eyebrow, title, copy, action }: { eyebrow: string; title: string; copy: string; action?: React.ReactNode }) {
  return (
    <div className="page-heading mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div><p className="page-eyebrow mb-2">{eyebrow}</p><h1 className="text-2xl font-semibold tracking-[-0.035em] sm:text-[2rem]">{title}</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">{copy}</p></div>
      {action}
    </div>
  );
}

function WelcomeTour({
  open,
  data,
  onFinish,
  onExperiment,
  onDemo,
}: {
  open: boolean;
  data: Bootstrap;
  onFinish: () => void;
  onExperiment: () => void;
  onDemo: () => void;
}) {
  const [step, setStep] = useState(0);
  useEffect(() => { if (open) setStep(0); }, [open]);
  const totalRecords = data.workspace.counts.runs + data.workspace.counts.reports;

  function finish(startExperiment = false) {
    onFinish();
    if (startExperiment) onExperiment();
  }

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!next) finish(); }}>
      <DialogContent className="welcome-dialog overflow-hidden p-0 sm:max-w-3xl">
        <div className="welcome-canvas min-h-[31rem] p-6 sm:p-9">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img className="size-11 object-contain" src="/p2p-logo-mark-dark.png" alt="" />
              <div><p className="text-sm font-semibold">P2P Studio</p><p className="text-xs text-muted-foreground">A local workspace for proxy population research</p></div>
            </div>
            <button type="button" className="rounded-lg px-2 py-1 text-sm text-muted-foreground hover:bg-background/70 hover:text-foreground" onClick={() => finish()}>Skip</button>
          </div>

          <div className="flex min-h-[21rem] items-center py-8" aria-live="polite">
            {step === 0 ? (
              <div className="mx-auto max-w-xl text-center">
                <span className="welcome-orbit mx-auto mb-7 flex size-28 items-center justify-center rounded-full"><FlaskConical className="size-11" /></span>
                <p className="page-eyebrow mb-3 justify-center">Welcome to your workspace</p>
                <h2 className="text-3xl font-semibold tracking-[-.04em] sm:text-4xl">From survey questions to a tested proxy ensemble.</h2>
                <p className="mx-auto mt-4 max-w-lg text-base leading-7 text-muted-foreground">Build experiments, follow local runs, and return to every result your research has produced—without sending workspace data to a hosted service.</p>
              </div>
            ) : step === 1 ? (
              <div className="w-full">
                <div className="mx-auto mb-7 max-w-xl text-center"><p className="page-eyebrow mb-3 justify-center">One continuous workflow</p><h2 className="text-3xl font-semibold tracking-[-.04em]">Each stage leaves an inspectable record.</h2></div>
                <div className="grid gap-3 sm:grid-cols-3">
                  {[
                    { label: "Stage 1", title: "Generate", copy: "Create a diverse proxy population from your survey and backend.", icon: Boxes, tone: "train" },
                    { label: "Stage 2", title: "Fit", copy: "Choose a method and learn how proxy responses combine.", icon: Gauge, tone: "valid" },
                    { label: "Stage 3", title: "Evaluate", copy: "Inspect held-out metrics, predictions, warnings, and provenance.", icon: BarChart3, tone: "test" },
                  ].map(({ label, title, copy, icon: Icon, tone }) => <div key={label} className={`welcome-stage welcome-stage-${tone}`}><div className="mb-5 flex items-center justify-between"><span className="flex size-10 items-center justify-center rounded-xl bg-current/10"><Icon className="size-5" /></span><span className="font-mono text-xs font-semibold">{label}</span></div><h3 className="text-lg font-semibold text-foreground">{title}</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">{copy}</p></div>)}
                </div>
              </div>
            ) : (
              <div className="w-full">
                <div className="mx-auto mb-7 max-w-xl text-center"><p className="page-eyebrow mb-3 justify-center">Start where you are</p><h2 className="text-3xl font-semibold tracking-[-.04em]">New work and past work belong together.</h2><p className="mt-3 text-base leading-7 text-muted-foreground">The same library reads runs in the current format and in the earlier format. Nothing is migrated or rewritten just to be viewed.</p></div>
                <div className="mx-auto grid max-w-xl grid-cols-3 overflow-hidden rounded-2xl border bg-background/75 text-center shadow-sm">
                  <div className="border-r p-4"><p className="font-mono text-2xl font-semibold">{data.workspace.counts.currentRuns}</p><p className="mt-1 text-xs text-muted-foreground">Current runs</p></div>
                  <div className="border-r p-4"><p className="font-mono text-2xl font-semibold">{data.workspace.counts.historicalRuns}</p><p className="mt-1 text-xs text-muted-foreground">Earlier-format runs</p></div>
                  <div className="p-4"><p className="font-mono text-2xl font-semibold">{data.workspace.counts.reports}</p><p className="mt-1 text-xs text-muted-foreground">Reports</p></div>
                </div>
                <p className="mt-4 text-center text-sm text-muted-foreground">{totalRecords ? `${totalRecords} records are ready in this workspace.` : "This workspace is new. Your first records will appear here automatically."}</p>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between gap-4 border-t border-border/70 pt-5">
            <div className="flex gap-1.5" aria-label={`Welcome step ${step + 1} of 3`}>{[0, 1, 2].map((index) => <span key={index} className={`h-1.5 rounded-full transition-all ${step === index ? "w-7 bg-primary" : "w-1.5 bg-border"}`} />)}</div>
            <div className="flex gap-2">
              {step > 0 && <Button variant="ghost" onClick={() => setStep((current) => current - 1)}>Back</Button>}
              {step < 2 ? <Button onClick={() => setStep((current) => current + 1)}>Continue <ChevronRight /></Button> : <><Button variant="ghost" onClick={() => { finish(); onDemo(); }}><Play /> Run offline demo</Button><Button variant="outline" onClick={() => finish()}>Open workspace</Button><Button onClick={() => finish(true)}><FlaskConical /> New experiment</Button></>}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function WorkspaceHome({
  data,
  onExperiment,
  onRuns,
  onRun,
  onReport,
  onJobs,
  onData,
  onDemo,
}: {
  data: Bootstrap;
  onExperiment: () => void;
  onRuns: () => void;
  onRun: (path: string) => void;
  onReport: (path: string) => void;
  onJobs: () => void;
  onData: () => void;
  onDemo: () => void;
}) {
  const activeJobs = data.jobs.filter((job) => ["running", "paused"].includes(job.status));
  const recoverableJobs = data.jobs.filter((job) => job.status === "interrupted");
  const hasRecords = data.workspace.counts.runs > 0 || data.workspace.counts.reports > 0;
  const recentRuns = data.runs.slice(0, 7);
  const recentReports = data.reports.slice(0, 3);
  const latestRun = recentRuns[0];
  const latestRunPath = latestRun?.path;
  const [previewQuestions, setPreviewQuestions] = useState<PredictionQuestion[] | null>(null);
  useEffect(() => {
    let active = true;
    setPreviewQuestions(null);
    if (!latestRunPath) return () => { active = false; };
    void api<RunDetail>(`/api/run?path=${encodeURIComponent(latestRunPath)}`)
      .then((detail) => {
        if (!active) return;
        const available = detail.questions || [];
        const heldOut = available.filter((question) => question.split?.toLowerCase() === "test");
        setPreviewQuestions(heldOut.length ? heldOut : available);
      })
      .catch(() => { if (active) setPreviewQuestions([]); });
    return () => { active = false; };
  }, [latestRunPath]);
  const latestDisplayName = latestRun ? friendlyRunName(latestRun.runId) : "";
  const latestWave = latestDisplayName.match(/(?:^| · )(Wave \d+)$/i)?.[1];
  const latestHeadline = latestDisplayName
    .replace(/^Stage 2 ablation · /i, "")
    .replace(/ · Wave \d+$/i, "");
  const workspaceInventory = (
    <section className={`workspace-status mb-5 grid overflow-hidden rounded-xl border sm:grid-cols-2 ${data.workspace.counts.historicalRuns > 0 || data.workspace.counts.reports > 0 ? "xl:grid-cols-4" : ""}`} aria-label="Workspace inventory">
      {[
        { label: "Current runs", value: data.workspace.counts.currentRuns, note: "Canonical format", icon: Activity, tone: "current" },
        { label: "Earlier-format runs", value: data.workspace.counts.historicalRuns, note: "Read-only", icon: History, tone: "historical" },
        { label: "Result folders", value: data.workspace.counts.reports, note: "Markdown summaries and figures", icon: FileText, tone: "reports" },
        { label: "Runnable manifests", value: data.workspace.counts.manifests, note: `${data.capabilities.backends.length} backends registered`, icon: FileCode2, tone: "manifests" },
      ].filter((card) => card.tone !== "historical" && card.tone !== "reports" || card.value > 0).map(({ label, value, note, icon: Icon, tone }) => <button key={label} type="button" className="workspace-stat group" onClick={label === "Runnable manifests" ? onExperiment : onRuns}><span className={`workspace-stat-icon workspace-stat-${tone}`}><Icon className="size-4" /></span><span className="min-w-0"><span className="block text-xs font-semibold uppercase tracking-[.075em] text-muted-foreground">{label}</span><span className="mt-1 flex items-baseline gap-2"><strong className="font-mono text-2xl tracking-tight">{value}</strong><span className="truncate text-xs text-muted-foreground">{note}</span></span></span><ChevronRight className="ml-auto size-4 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" /></button>)}
    </section>
  );

  return (
    <>
      <PageHeading
        eyebrow={`Workspace / ${data.repository}`}
        title={hasRecords ? "Continue your research" : "Your P2P workspace is ready"}
        copy={hasRecords ? "Resume recent work, inspect the full research record, or begin another experiment." : "Connect local data and create a first experiment. Studio will keep its runs, reports, and provenance together."}
        action={<Button onClick={onExperiment}><FlaskConical /> New experiment</Button>}
      />

      {!hasRecords ? (
        <><div className="grid gap-5 lg:grid-cols-3">
          {[
            { step: "01", title: "Run the offline demo", copy: "Run the whole pipeline on a fictional survey: simulated residents (rules, not a language model) are generated and surveyed, then Stage 2 is fitted and evaluated. No credentials, network or cost.", icon: Play, action: onDemo, label: "Run demo" },
            { step: "02", title: "Add your data", copy: "Inspect a survey, map its columns, verify splits, and check target coverage before registering it.", icon: Database, action: onData, label: "Inspect local data" },
            { step: "03", title: "Build an experiment", copy: "Choose an intent, use backend-provided options, and pass preflight before execution.", icon: FlaskConical, action: onExperiment, label: "New experiment" },
          ].map(({ step, title, copy, icon: Icon, action, label }) => <section key={step} className="surface-card p-5 sm:p-6"><div className="flex items-center justify-between"><span className="step-index">{step}</span><Icon className="size-5 text-primary" /></div><h2 className="mt-6 text-lg font-semibold">{title}</h2><p className="mt-2 min-h-12 text-sm leading-6 text-muted-foreground">{copy}</p><Button variant="ghost" className="mt-4 -ml-3" onClick={action}>{label} <ChevronRight /></Button></section>)}
        </div></>
      ) : (
        <div className="min-w-0">
          {latestRun && <section className="research-focus-panel research-focus-hero mb-5">
            <div className="research-focus-heading"><p className="page-eyebrow">Latest evidence</p><h2>{latestHeadline || latestDisplayName}</h2><p>{latestDisplayName.startsWith("Stage 2 ablation") ? "Stage 2 ablation · " : ""}{latestWave ? `${latestWave} · ` : ""}{latestRun.origin === "legacy" ? "Earlier-format run" : "Current pipeline"} · {formatDate(latestRun.modifiedAt)}</p><code title={latestRun.path}>{latestRun.runId}</code></div>
            <div className="research-focus-visuals"><PopulationMiniature population={latestRun.population} /><HeldOutPredictionPreview questions={previewQuestions} /></div>
            <div className="research-focus-evidence" aria-label="Latest run ensemble evidence"><p className="page-eyebrow">Ensemble evidence</p><dl><div><dt>Selected proxies</dt><dd>{formatNumber(latestRun.population?.selected)}</dd></div><div><dt>Method</dt><dd>{latestRun.method || "—"}</dd></div><div><dt>Held-out questions</dt><dd>{previewQuestions === null ? "…" : formatNumber(previewQuestions.length)}</dd></div><div><dt>1 − JSD</dt><dd>{formatNumber(runScore(latestRun), 3)}</dd></div></dl><p>Dimmed proxies were generated but not retained. Stripes identify negative coefficients.</p></div>
            <div className="research-focus-actions"><button type="button" className="research-focus-open-result" onClick={() => onRun(latestRun.path)}><span><small>Research record</small><strong>Open complete result</strong></span><i><ChevronRight /></i></button></div>
          </section>}

          {workspaceInventory}

          <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1.5fr)_340px]">
          <section className="surface-card min-w-0 overflow-hidden">
            <div className="home-recent-header flex items-center justify-between border-b px-5 py-4 sm:px-6"><div className="min-w-0"><h2 className="section-title">Recent runs</h2><p className="section-copy">Newest first, across both formats.</p></div><Button variant="ghost" size="sm" onClick={onRuns}>Full library <ChevronRight /></Button></div>
            <div className="hidden grid-cols-[minmax(240px,1fr)_110px_90px_110px] gap-4 border-b bg-muted/40 px-6 py-2.5 text-xs font-semibold uppercase tracking-[.065em] text-muted-foreground md:grid"><span>Run</span><span>Method</span><span>1 − JSD</span><span>Updated</span></div>
            {recentRuns.map((run) => <button key={run.path} className="run-row grid w-full gap-2 border-b px-5 py-3.5 text-left last:border-0 md:grid-cols-[minmax(240px,1fr)_110px_90px_110px] md:items-center md:gap-4 sm:px-6" onClick={() => onRun(run.path)}><span className="min-w-0"><span className="flex min-w-0 items-center gap-2"><span className={`record-signal ${run.origin === "legacy" ? "record-signal-historical" : "record-signal-current"}`} /><strong className="truncate text-sm">{run.runId}</strong>{run.origin === "legacy" && <Badge variant="outline" className="shrink-0 text-[10px]">earlier format</Badge>}</span><span className="mt-1 block truncate pl-4 font-mono text-[11px] text-muted-foreground">{run.path}</span></span><span className="text-sm text-muted-foreground">{run.method || "Unavailable"}</span><span className="font-mono text-sm font-semibold">{formatNumber(runScore(run), 3)}</span><span className="text-xs text-muted-foreground">{formatDate(run.modifiedAt)}</span></button>)}
          </section>

          <div className="min-w-0 space-y-5">
            {recoverableJobs.length > 0 && <section className="surface-card overflow-hidden"><div className="border-b px-5 py-4"><div className="flex items-center gap-2"><TriangleAlert className="size-4 text-amber-600" /><h2 className="section-title">Interrupted work</h2></div><p className="section-copy">Durable records recovered after Studio restarted.</p></div>{recoverableJobs.slice(0, 3).map((job) => <button key={job.id} type="button" onClick={onJobs} className="flex w-full items-center gap-3 border-b px-5 py-3.5 text-left last:border-0 hover:bg-muted/45"><span className="min-w-0 flex-1"><strong className="block truncate text-sm">{job.output}</strong><span className="mt-0.5 block text-xs text-muted-foreground">{job.recovery === "checkpoint" ? "Checkpoint detected; verify the resolved config before resuming." : "Inspect partial artifacts or choose a new output directory."}</span></span><Badge variant="outline">interrupted</Badge></button>)}</section>}

            {recentReports.length > 0 && <section className="surface-card overflow-hidden">
              <div className="flex items-center justify-between border-b px-5 py-4"><div><h2 className="section-title">Recent reports</h2><p className="section-copy">Readable research records.</p></div><FileText className="size-4 text-muted-foreground" /></div>
              {recentReports.length ? recentReports.map((report) => <button key={report.path} className="flex w-full items-center gap-3 border-b px-5 py-3.5 text-left last:border-0 hover:bg-muted/45" onClick={() => onReport(report.path)}><span className="report-glyph"><FileText className="size-4" /></span><span className="min-w-0 flex-1"><strong className="block truncate text-sm">{report.name}</strong><span className="mt-0.5 block truncate text-xs text-muted-foreground">{report.title}</span></span><ChevronRight className="size-4 shrink-0 text-muted-foreground" /></button>) : <div className="p-6 text-sm leading-6 text-muted-foreground">No reports have been written yet.</div>}
            </section>}

            <button type="button" onClick={onJobs} className="flex w-full items-center justify-between rounded-xl border bg-card px-4 py-3 text-left text-sm hover:bg-muted/45"><span className="flex items-center gap-2 text-muted-foreground"><span className={`size-2 rounded-full ${activeJobs.length ? "bg-amber-500" : "bg-emerald-500"}`} />{activeJobs.length ? `${activeJobs.length} active ${activeJobs.length === 1 ? "job" : "jobs"}` : "No active jobs"}</span><span className="text-xs text-muted-foreground">Latest {latestRun ? formatDate(latestRun.modifiedAt) : "—"}</span></button>
          </div>
          </div>
        </div>
      )}
    </>
  );
}

type BuilderStepState = { id: string; title: string; done: boolean };

// The builder's steps, in workflow order. Each shows whether it is complete and
// jumps to its section; the step in view is marked as current.
function StepRail({ steps }: { steps: BuilderStepState[] }) {
  const [current, setCurrent] = useState(steps[0]?.id);
  const ids = steps.map((step) => step.id).join("|");
  useEffect(() => {
    const sections = steps.map((step) => document.getElementById(`builder-step-${step.id}`)).filter((item): item is HTMLElement => Boolean(item));
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((left, right) => left.boundingClientRect.top - right.boundingClientRect.top)[0];
      if (visible) setCurrent(visible.target.id.replace("builder-step-", ""));
    }, { rootMargin: "-30% 0px -55% 0px" });
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [ids]);
  return (
    <nav className="step-rail" aria-label="Experiment steps">
      <ol>
        {steps.map((step, index) => (
          <li key={step.id}>
            <button type="button" className={`step-rail-item${current === step.id ? " step-rail-current" : ""}${step.done ? " step-rail-done" : ""}`} aria-current={current === step.id ? "step" : undefined} onClick={() => document.getElementById(`builder-step-${step.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}>
              <span className="step-rail-spot" aria-hidden="true" />
              <span className="step-rail-number">{String(index + 1).padStart(2, "0")}</span>
              <span className="step-rail-title">{step.title}</span>
              <span className="sr-only">{step.done ? "complete" : "incomplete"}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

function GridInputs({ id, label, value, onChange, hint }: { id: string; label: string; value: [number, number, number]; onChange: (value: [number, number, number]) => void; hint: string }) {
  const parts = ["start", "stop", "count"] as const;
  return <div className="field-stack sm:col-span-2"><Label htmlFor={`${id}-start`}>{label}</Label><div className="grid grid-cols-3 gap-2">{parts.map((part, index) => <div key={part}><Input id={`${id}-${part}`} aria-label={`${label} ${part}`} type="number" step={part === "count" ? 1 : "any"} min={part === "count" ? 1 : undefined} value={value[index]} onChange={(event) => { const next = [...value] as [number, number, number]; next[index] = Number(event.target.value); onChange(next); }} className="h-10 bg-background font-mono text-xs" /><p className="mt-1 text-[11px] text-muted-foreground">{part}</p></div>)}</div><p className="text-xs leading-5 text-muted-foreground">{hint}</p></div>;
}

function Stage2Controls({ draft, setField, caps, advancedOpen, usesSolver }: {
  draft: ExperimentDraft;
  setField: <K extends keyof ExperimentDraft>(key: K, value: ExperimentDraft[K]) => void;
  caps: Capabilities;
  advancedOpen: boolean;
  usesSolver: boolean;
}) {
  const stage2 = caps.stage2;
  const reanchored = Boolean(draft.referenceDistribution.trim());
  const methods = caps.models.filter((model) => model.guided && !(reanchored && stage2.reference.excludedMethods.includes(model.id)));
  const selected = methods.find((model) => model.id === draft.modelMethod);
  const configurations = reanchored ? referenceConfigurations(caps) : stage2.configurations;
  const configuration = configurations.find((item) => item.id === draft.configuration);
  const fitsPenalty = draft.modelMethod !== SIMPLE_AVERAGE;
  const methodZeroThreshold = selected?.zeroThreshold ?? stage2.zeroThreshold;
  const [low, high, count] = draft.alphaGrid;
  return <>
    <div className="field-stack"><Label>Stage 2 method</Label><Select value={draft.modelMethod} onValueChange={(value) => setField("modelMethod", value)}><SelectTrigger className="h-10 w-full bg-background"><SelectValue /></SelectTrigger><SelectContent>{methods.map((model) => <SelectItem key={model.id} value={model.id}>{model.label}</SelectItem>)}</SelectContent></Select><p className="text-xs leading-5 text-muted-foreground">{selected?.description || "From the live Python model registry."}</p></div>
    {fitsPenalty ? <div className="field-stack"><Label>Stage-2 configuration</Label><Select value={draft.configuration} onValueChange={(value) => setField("configuration", value)}><SelectTrigger className="h-10 w-full bg-background"><SelectValue placeholder="Choose a configuration" /></SelectTrigger><SelectContent>{configurations.map((item) => <SelectItem key={item.id} value={item.id}>{item.label}</SelectItem>)}</SelectContent></Select><p className="text-xs leading-5 text-muted-foreground">{configuration?.description || "This configuration cannot be combined with a reference distribution; choose another."}{reanchored && configuration ? " Re-anchored: a reference respondent or an intercept would change what the reference's weight of one means, so they are not offered." : ""}</p></div>
      : <div className="rounded-xl border bg-muted/30 p-4 text-xs leading-5 text-muted-foreground sm:col-span-1">Every agent gets the same weight. There is no penalty to select, so penalty, validation and configuration settings do not apply; evaluation is the same as for fitted methods.</div>}
    {fitsPenalty && !advancedOpen && <p className="text-xs leading-5 text-muted-foreground sm:col-span-2">Penalty chosen by {draft.validationStrategy === "holdout" ? `holdout on the ${draft.holdoutSplit} split` : `${draft.cvFolds}-fold cross-validation grouped by question`} over α from 10<sup>{low}</sup> to 10<sup>{high}</sup> ({count} values). Open advanced settings to change validation, folds, grids, weighting or the solver.</p>}
    {fitsPenalty && advancedOpen && <>
      <div className="field-stack"><Label>Penalty selection</Label><select aria-label="Penalty selection" value={draft.validationStrategy} onChange={(event) => setField("validationStrategy", event.target.value as ExperimentDraft["validationStrategy"])} className="h-10 rounded-md border bg-background px-3 text-sm">{stage2.validationStrategies.map((strategy) => <option key={strategy} value={strategy}>{strategy === "cv" ? "Cross-validation (grouped by question)" : "Holdout split (no refit)"}</option>)}</select></div>
      {draft.validationStrategy === "cv" ? <>
        <div className="field-stack"><Label htmlFor="cv-folds">Cross-validation folds</Label><Input id="cv-folds" type="number" min={2} value={draft.cvFolds} onChange={(event) => setField("cvFolds", Number(event.target.value))} className="h-10 bg-background" /></div>
        <div className="field-stack"><Label>Fold assignment</Label><select aria-label="Fold assignment" value={draft.foldMethod} onChange={(event) => setField("foldMethod", event.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm">{stage2.foldMethods.map((method) => <option key={method} value={method}>{method === "hashed" ? "Hashed (default; same on every machine)" : method === "sklearn" ? "scikit-learn GroupKFold (machine-dependent)" : "Replay a recorded folds.json"}</option>)}</select></div>
        {draft.foldMethod === "replay" && <div className="field-stack"><Label htmlFor="fold-file">Fold record</Label><Input id="fold-file" value={draft.foldFile} onChange={(event) => setField("foldFile", event.target.value)} placeholder="outputs/.../folds.json" className="h-10 bg-background font-mono text-xs" /></div>}
      </> : <div className="field-stack"><Label htmlFor="holdout-split">Holdout split</Label><Input id="holdout-split" value={draft.holdoutSplit} onChange={(event) => setField("holdoutSplit", event.target.value)} className="h-10 bg-background font-mono text-xs" /><p className="text-xs leading-5 text-muted-foreground">Each penalty is fitted on the other training splits and scored on this one; the best fit is kept without refitting.</p></div>}
      <GridInputs id="alpha-grid" label="Penalty grid α (log10 start, log10 stop, count)" value={draft.alphaGrid} onChange={(value) => setField("alphaGrid", value)} hint={`P2P's default is 10^${stage2.alphaGrid.logspace[0]} to 10^${stage2.alphaGrid.logspace[1]} with ${stage2.alphaGrid.logspace[2]} values, wide enough that its ceiling is rarely selected. The run reports a selection at either edge.`} />
      {draft.modelMethod === "elasticnet" && <GridInputs id="l1-grid" label="l1_ratio grid (start, stop, count)" value={draft.l1RatioGrid} onChange={(value) => setField("l1RatioGrid", value)} hint="1 is pure lasso; values towards 0 add more ridge penalty. Must lie between 0 and 1." />}
      <label className="flex items-start gap-3 rounded-xl border bg-background p-3 text-sm sm:col-span-2"><input type="checkbox" className="mt-1 size-4 accent-[var(--primary)]" checked={draft.exactQuestionMass} onChange={(event) => setField("exactQuestionMass", event.target.checked)} /><span><strong className="block font-medium">Exact question mass</strong><span className="text-xs leading-5 text-muted-foreground">Each question's prediction sums exactly to one (soft_group_sum: false), replacing the soft mass penalty, so the configuration's penalty strength no longer matters.{reanchored ? " With a reference distribution, the agents' weights then sum to zero." : ""}</span></span></label>
      <div className="field-stack"><Label htmlFor="zero-threshold">Zero threshold</Label><Input id="zero-threshold" inputMode="decimal" value={draft.zeroThreshold} onChange={(event) => setField("zeroThreshold", event.target.value)} placeholder={String(methodZeroThreshold)} className="h-10 bg-background font-mono text-xs" /><p className="text-xs leading-5 text-muted-foreground">Fitted weights smaller than this are set to zero. Leave empty for {methodZeroThreshold === stage2.zeroThreshold ? `P2P's default (${stage2.zeroThreshold})` : `this method's default (${methodZeroThreshold})`}.</p></div>
      <label className="flex items-start gap-3 rounded-xl border bg-background p-3 text-sm sm:col-span-2"><input type="checkbox" className="mt-1 size-4 accent-[var(--primary)]" checked={draft.blockWeighting} onChange={(event) => setField("blockWeighting", event.target.checked)} /><span><strong className="block font-medium">Weight each question equally</strong><span className="text-xs leading-5 text-muted-foreground">Block weighting: every question contributes the same total weight to the fit, whatever its number of options.</span></span></label>
      {usesSolver && <div className="field-stack sm:col-span-2"><Label>Solver</Label><select aria-label="Solver" value={draft.solver} onChange={(event) => setField("solver", event.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm"><option value="">{stage2.defaultSolver} (P2P default)</option>{stage2.solvers.filter((item) => item !== stage2.defaultSolver).map((item) => <option key={item} value={item}>{item}</option>)}</select><p className="text-xs leading-5 text-muted-foreground">If a solve fails, P2P falls back in the order {stage2.solverFallbacks.join(" → ")} and records which solver produced the fit. Choose OSQP to reproduce fits made with it.</p></div>}
    </>}
  </>;
}

function MetricChooser({ draft, setField, caps }: { draft: ExperimentDraft; setField: <K extends keyof ExperimentDraft>(key: K, value: ExperimentDraft[K]) => void; caps: Capabilities }) {
  const toggle = (id: string) => setField("metrics", draft.metrics.includes(id) ? draft.metrics.filter((item) => item !== id) : [...draft.metrics, id]);
  return <div className="field-stack sm:col-span-2"><Label>Metrics</Label><div className="grid gap-2 sm:grid-cols-2">{caps.metrics.map((metric) => <label key={metric.id} className="flex items-start gap-3 rounded-xl border bg-background p-3 text-sm" title={metric.description}><input type="checkbox" className="mt-1 size-4 accent-[var(--primary)]" checked={draft.metrics.includes(metric.id)} onChange={() => toggle(metric.id)} /><span><strong className="block font-medium">{metric.label} <span className="font-normal text-muted-foreground">{metric.direction === "lower" ? "↓" : "↑"}</span></strong><span className="text-xs leading-5 text-muted-foreground">{metric.description}</span></span></label>)}</div>{!draft.metrics.length && <p className="text-xs text-rose-700">Choose at least one metric.</p>}</div>;
}

function EvaluationChooser({ draft, setField, caps }: { draft: ExperimentDraft; setField: <K extends keyof ExperimentDraft>(key: K, value: ExperimentDraft[K]) => void; caps: Capabilities }) {
  const policy = caps.evaluation.distributionPolicies.find((item) => item.id === draft.distributionPolicy);
  const invalid = caps.evaluation.invalidDistribution.find((item) => item.id === draft.invalidDistribution);
  return <>
    <div className="field-stack"><Label>Distribution policy</Label><select aria-label="Distribution policy" value={draft.distributionPolicy} onChange={(event) => setField("distributionPolicy", event.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm">{caps.evaluation.distributionPolicies.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select><p className="text-xs leading-5 text-muted-foreground">{policy?.description} Used by rescaled MSE, 1 − JSD, EMD and MCPA; raw MSE always scores the raw predictions.</p></div>
    <div className="field-stack"><Label>When a prediction is still invalid</Label><select aria-label="Invalid prediction handling" value={draft.invalidDistribution} onChange={(event) => setField("invalidDistribution", event.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm">{caps.evaluation.invalidDistribution.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select><p className="text-xs leading-5 text-muted-foreground">{invalid?.description}</p></div>
  </>;
}

function NewExperiment({ data, onRefresh, sourceManifestPath }: { data: Bootstrap; onRefresh: () => Promise<void>; sourceManifestPath?: string | null }) {
  const defaults = data.capabilities.defaults;
  const [draft, setDraft] = useState<ExperimentDraft>(() => {
    const initial: ExperimentDraft = {
      command: "pipeline",
      name: `studio-${new Date().toISOString().slice(0, 10)}`,
      seed: defaults.seed,
      surveySource: "",
      surveySchema: data.surveys[0]?.path || "",
      target: "",
      targetSubgroup: "",
      referenceDistribution: "",
      basis: "",
      predictions: "",
      generationMode: defaults.generationMode,
      targetN: defaults.targetN,
      initialN: 10,
      updateSteps: 10,
      attributeBank: "config/attribute_banks/attribute_bank.yaml",
      backendProfile: data.backends.find((item) => item.local)?.path || data.backends[0]?.path || "",
      modelMethod: defaults.modelMethod || "lasso",
      cvFolds: defaults.cvFolds,
      configuration: data.capabilities.stage2.defaultConfiguration,
      validationStrategy: "cv",
      holdoutSplit: defaults.holdoutSplit,
      foldMethod: data.capabilities.stage2.defaultFoldMethod,
      foldFile: "",
      blockWeighting: false,
      exactQuestionMass: false,
      zeroThreshold: "",
      solver: "",
      alphaGrid: data.capabilities.stage2.alphaGrid.logspace,
      l1RatioGrid: data.capabilities.stage2.l1RatioGrid.linspace,
      metrics: defaults.metrics,
      distributionPolicy: defaults.distributionPredictionPolicy,
      invalidDistribution: defaults.invalidDistribution,
    };
    try {
      const saved = JSON.parse(window.localStorage.getItem(draftPreferenceKey) || "null");
      return saved && typeof saved === "object" ? { ...initial, ...saved } : initial;
    } catch { return initial; }
  });
  const [selectedDataset, setSelectedDataset] = useState("manual");
  const [loadedSource, setLoadedSource] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [activeJob, setActiveJob] = useState<Job | null>(data.jobs.find((job) => ["running", "paused"].includes(job.status)) || null);
  const [starting, setStarting] = useState(false);
  const [createdManifest, setCreatedManifest] = useState<CreatedManifest | null>(null);
  const [preflight, setPreflight] = useState<ManifestPreview | null>(null);
  // The subgroups preflight last found in a target file, kept while that file is chosen.
  const [knownSubgroups, setKnownSubgroups] = useState<{ target: string; groups: NonNullable<ManifestPreview["targetSubgroups"]> } | null>(null);
  const targetSubgroups = knownSubgroups?.target === draft.target.trim() ? knownSubgroups.groups : null;
  const config = useMemo(() => buildGuidedConfig(draft, data.capabilities), [draft, data.capabilities]);
  const yamlPreview = useMemo(() => `${yamlLines(config).join("\n")}\n`, [config]);
  const manifestPath = `config/experiments/studio/${slug(draft.name) || "new-experiment"}.yaml`;
  const output = `outputs/studio/${slug(draft.name) || "new-run"}`;
  const selectedBackend = data.backends.find((item) => item.path === draft.backendProfile);
  const needsGeneration = ["pipeline", "generate"].includes(draft.command);
  const needsTarget = ["pipeline", "regress", "evaluate"].includes(draft.command);
  const needsModel = ["pipeline", "regress"].includes(draft.command);
  const requiredPaths = [draft.surveySource, draft.surveySchema, needsTarget ? draft.target : "ready", draft.command === "regress" ? draft.basis : "ready", draft.command === "evaluate" ? draft.predictions : "ready", needsGeneration ? draft.backendProfile : "ready", needsGeneration && draft.generationMode === "active" ? draft.attributeBank : "ready"];
  const fitsPenalty = needsModel && draft.modelMethod !== SIMPLE_AVERAGE;
  const usesSolver = data.capabilities.stage2.solverCommands.includes(draft.command);
  const gridOk = (grid: [number, number, number]) => grid.every((value) => Number.isFinite(value)) && Number.isInteger(grid[2]) && grid[2] > 0 && grid[1] >= grid[0];
  const reanchored = needsModel && Boolean(draft.referenceDistribution.trim());
  const referenceReady = !reanchored || (
    !data.capabilities.stage2.reference.excludedMethods.includes(draft.modelMethod)
    && referenceConfigurations(data.capabilities).some((item) => item.id === draft.configuration)
  );
  const zeroThresholdOk = !draft.zeroThreshold.trim() || (Number.isFinite(Number(draft.zeroThreshold)) && Number(draft.zeroThreshold) >= 0);
  const stage2Ready = referenceReady && zeroThresholdOk && (!fitsPenalty || (
    gridOk(draft.alphaGrid)
    && (draft.modelMethod !== "elasticnet" || (gridOk(draft.l1RatioGrid) && draft.l1RatioGrid[0] >= 0 && draft.l1RatioGrid[1] <= 1))
    && (draft.validationStrategy === "holdout" ? Boolean(draft.holdoutSplit.trim()) : draft.cvFolds > 1 && (draft.foldMethod !== "replay" || Boolean(draft.foldFile.trim())))
  ));
  const ready = Boolean(draft.name.trim()) && requiredPaths.every((value) => value.trim()) && draft.seed >= 0 && (!needsGeneration || draft.targetN > 0) && stage2Ready && (!needsTarget || draft.metrics.length > 0);
  const solverFlag = usesSolver && draft.solver ? ` --solver ${draft.solver}` : "";
  const dataPaths = [draft.surveySource, draft.surveySchema, needsTarget ? draft.target : "ready", draft.command === "regress" ? draft.basis : "ready", draft.command === "evaluate" ? draft.predictions : "ready"];
  const steps: BuilderStepState[] = [
    { id: "identity", title: "Experiment", done: Boolean(draft.name.trim()) && draft.seed >= 0 },
    { id: "data", title: "Data", done: dataPaths.every((value) => value.trim()) },
    ...(needsGeneration ? [{ id: "stage1", title: "Stage 1", done: Boolean(draft.backendProfile.trim()) && draft.targetN > 0 && (draft.generationMode !== "active" || Boolean(draft.attributeBank.trim())) }] : []),
    ...(needsModel ? [{ id: "stage2", title: "Stage 2", done: stage2Ready }] : []),
    ...(needsTarget ? [{ id: "evaluation", title: "Evaluation", done: draft.metrics.length > 0 }] : []),
    { id: "review", title: "Review", done: Boolean(preflight?.valid) },
  ];
  const stepNumber = (id: string) => String(steps.findIndex((step) => step.id === id) + 1).padStart(2, "0");
  const selectedModel = data.capabilities.models.find((model) => model.id === draft.modelMethod);
  const selectedConfiguration = data.capabilities.stage2.configurations.find((item) => item.id === draft.configuration);
  const setField = <K extends keyof ExperimentDraft>(key: K, value: ExperimentDraft[K]) => setDraft((current) => ({ ...current, [key]: value }));
  useEffect(() => { setPreflight(null); }, [yamlPreview, output, draft.solver]);
  useEffect(() => { window.localStorage.setItem(draftPreferenceKey, JSON.stringify(draft)); }, [draft]);
  useEffect(() => {
    if (!sourceManifestPath || loadedSource === sourceManifestPath) return;
    void api<ManifestDetail>(`/api/manifest?path=${encodeURIComponent(sourceManifestPath)}`).then((detail) => {
      const config = detail.config;
      const run = config.run || {};
      const inputs = config.data || {};
      const survey = inputs.survey || {};
      const generation = config.generation || {};
      const model = config.model || {};
      const validation = model.validation || {};
      const folds = validation.folds || {};
      const backend = config.backend || {};
      const stage2 = data.capabilities.stage2;
      setDraft((current) => ({
        ...current,
        command: detail.command,
        name: `${run.name || detail.summary.name}-copy`,
        seed: Number(run.seed ?? current.seed),
        surveySource: String(survey.source || ""),
        surveySchema: String(survey.schema || ""),
        target: String((inputs.target && typeof inputs.target === "object" ? inputs.target.path : inputs.target) || ""),
        targetSubgroup: String((inputs.target && typeof inputs.target === "object" ? inputs.target.subgroup : "") || ""),
        referenceDistribution: String(inputs.reference_distribution || ""),
        basis: String(inputs.basis || ""),
        predictions: String(inputs.predictions || ""),
        generationMode: generation.mode === "vanilla" ? "vanilla" : "active",
        targetN: Number(generation.target_n ?? current.targetN),
        initialN: Number(generation.initial_n ?? current.initialN),
        updateSteps: Number(generation.num_update_steps ?? current.updateSteps),
        attributeBank: String(generation.attribute_bank?.source || current.attributeBank),
        backendProfile: String(backend.profile || current.backendProfile),
        modelMethod: guidedMethodOf(model, data.capabilities) || current.modelMethod,
        cvFolds: Number(validation.cv_folds ?? current.cvFolds),
        configuration: configurationOf(model, data.capabilities),
        validationStrategy: validation.strategy === "holdout" ? "holdout" : "cv",
        holdoutSplit: String(validation.holdout_split || current.holdoutSplit),
        foldMethod: String(folds.method || stage2.defaultFoldMethod),
        foldFile: String(folds.file || ""),
        blockWeighting: Boolean(model.block_weighting),
        exactQuestionMass: model.soft_group_sum === false,
        zeroThreshold: model.zero_threshold != null && Number(model.zero_threshold) !== (data.capabilities.models.find((item) => item.id === guidedMethodOf(model, data.capabilities))?.zeroThreshold ?? stage2.zeroThreshold) ? String(model.zero_threshold) : "",
        alphaGrid: gridSpec(model.alpha, "logspace", stage2.alphaGrid.logspace),
        l1RatioGrid: gridSpec(model.l1_ratio, "linspace", stage2.l1RatioGrid.linspace),
        metrics: Array.isArray(config.metrics) && config.metrics.length ? config.metrics.map(String) : current.metrics,
        // legacy_signed_normalize is the earlier spelling of signed_sum_normalize.
        distributionPolicy: String(config.evaluation?.distribution_prediction_policy || defaults.distributionPredictionPolicy).replace("legacy_signed_normalize", "signed_sum_normalize"),
        invalidDistribution: String(config.evaluation?.invalid_distribution || defaults.invalidDistribution),
      }));
      setSelectedDataset("manual");
      setLoadedSource(sourceManifestPath);
      toast.success(`Loaded ${detail.summary.name} as an editable copy.`);
    }).catch((error) => {
      setLoadedSource(sourceManifestPath);
      toast.error(error instanceof Error ? error.message : "Could not load the source manifest.");
    });
  }, [loadedSource, sourceManifestPath]);

  function chooseDataset(value: string) {
    setSelectedDataset(value);
    if (value === "manual") return;
    const dataset = data.datasets.find((item) => item.path === value);
    if (!dataset) return;
    setDraft((current) => ({
      ...current,
      surveySource: dataset.source || "",
      surveySchema: dataset.schema || "",
      target: dataset.target || "",
      targetSubgroup: "",
    }));
  }

  const refreshJob = useCallback(async () => {
    if (!activeJob) return;
    try {
      const payload = await api<{ jobs: Job[] }>("/api/jobs");
      const next = payload.jobs.find((job) => job.id === activeJob.id);
      if (next) setActiveJob(next);
    } catch { /* The page-level connection state handles server errors. */ }
  }, [activeJob?.id]);

  useEffect(() => {
    if (!activeJob || !["running", "paused"].includes(activeJob.status)) return;
    const timer = window.setInterval(() => void refreshJob(), 1500);
    return () => window.clearInterval(timer);
  }, [activeJob?.id, activeJob?.status, refreshJob]);

  async function start() {
    if (!ready || !preflight?.valid) return;
    if (preflight.paid && !window.confirm("This backend uses a metered provider and the current draft has no attached price estimate. Start it using your locally configured credentials?")) return;
    setStarting(true);
    try {
      const created = await api<CreatedManifest>("/api/manifests", { method: "POST", body: JSON.stringify({ command: draft.command, config }) });
      setCreatedManifest(created);
      const job = await api<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ command: draft.command, config: created.path, output, solver: usesSolver && draft.solver ? draft.solver : undefined }) });
      setActiveJob(job);
      await onRefresh();
      toast.success("YAML created and experiment started locally.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not create or start the experiment.");
    } finally { setStarting(false); }
  }

  async function reviewPreflight() {
    if (!ready) return;
    setStarting(true);
    try {
      const result = await api<ManifestPreview>("/api/manifests/preview", { method: "POST", body: JSON.stringify({ command: draft.command, config, output, solver: usesSolver && draft.solver ? draft.solver : undefined }) });
      setPreflight(result);
      setKnownSubgroups(result.targetSubgroups ? { target: draft.target.trim(), groups: result.targetSubgroups } : null);
      if (result.valid) toast.success("Preflight passed. Review the digest and warnings before running.");
      else toast.error("Preflight found a blocking conflict.");
    } catch (error) {
      setPreflight(null);
      toast.error(error instanceof Error ? error.message : "Preflight failed.");
    } finally { setStarting(false); }
  }

  async function act(action: "pause" | "resume" | "stop") {
    if (!activeJob) return;
    try {
      const job = await api<Job>(`/api/jobs/${activeJob.id}/action`, { method: "POST", body: JSON.stringify({ action }) });
      setActiveJob(job);
      toast.success(action === "pause" ? "Run paused." : action === "resume" ? "Run resumed." : "Run stopped.");
      if (action === "stop") await onRefresh();
    } catch (error) { toast.error(error instanceof Error ? error.message : "Action failed."); }
  }

  const backendLabel = selectedBackend?.modules.map((item) => item.model).filter((value, index, all) => all.indexOf(value) === index).join(", ") || selectedBackend?.name || "Not required";
  return (
    <>
      <PageHeading eyebrow="Experiments / New run" title="Build and run an experiment" copy="Describe the workflow step by step. Studio creates a validated YAML manifest under config/experiments/studio/ and then runs the same local P2P command as the CLI." action={<Button variant="outline" onClick={() => setAdvancedOpen((value) => !value)}>{advancedOpen ? "Hide advanced settings" : "Show advanced settings"}</Button>} />
      <StepRail steps={steps} />
      <div className="experiment-workspace grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-5">
          <section id="builder-step-identity" className="surface-card builder-step overflow-hidden">
            <div className="border-b border-border/70 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><span className="step-index">{stepNumber("identity")}</span><h2 className="section-title">Experiment</h2></div><p className="section-copy">Choose the workflow and name the manifest and output.</p></div>
            <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
              <div className="field-stack"><Label>Workflow</Label><Select value={draft.command} onValueChange={(value) => setField("command", value as GuidedCommand)}><SelectTrigger className="h-10 w-full bg-background"><SelectValue /></SelectTrigger><SelectContent>{data.capabilities.workflows.map((workflow) => <SelectItem key={workflow.id} value={workflow.id}>{workflow.label}</SelectItem>)}</SelectContent></Select></div>
              <div className="field-stack"><Label htmlFor="run-name">Experiment name</Label><Input id="run-name" value={draft.name} onChange={(event) => setField("name", event.target.value)} className="h-10 bg-background" /></div>
              {advancedOpen && <div className="field-stack"><Label htmlFor="run-seed">Random seed</Label><Input id="run-seed" type="number" min={0} value={draft.seed} onChange={(event) => setField("seed", Number(event.target.value))} className="h-10 bg-background" /></div>}
              <div className="field-stack"><Label>Manifest destination</Label><div className="flex h-10 items-center truncate rounded-md border bg-muted/35 px-3 font-mono text-xs">{manifestPath}</div></div>
            </div>
          </section>

          <section id="builder-step-data" className="surface-card builder-step overflow-hidden">
            <div className="border-b border-border/70 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><span className="step-index">{stepNumber("data")}</span><h2 className="section-title">Data</h2></div><p className="section-copy">Use repository-relative paths. They are validated before the YAML is saved.</p></div>
            <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
              <div className="field-stack sm:col-span-2"><Label>Registered dataset</Label><Select value={selectedDataset} onValueChange={chooseDataset}><SelectTrigger className="h-10 w-full bg-background"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="manual">Enter repository paths manually</SelectItem>{data.datasets.map((dataset) => <SelectItem key={dataset.path} value={dataset.path}>{dataset.name} · {dataset.questionCount ?? "—"} questions</SelectItem>)}</SelectContent></Select><p className="text-xs leading-5 text-muted-foreground">Registered descriptors reuse pinned local sources without copying or uploading the underlying data.</p></div>
              {selectedDataset !== "manual" ? <div className="sm:col-span-2 rounded-xl border bg-muted/30 p-4"><div className="flex items-start justify-between gap-4"><div className="min-w-0"><p className="text-sm font-semibold">Dataset paths applied</p><p className="mt-1 truncate font-mono text-xs text-muted-foreground">{draft.surveySource}</p><p className="mt-1 truncate font-mono text-xs text-muted-foreground">{draft.target || "No target attached"}</p></div><Button variant="outline" size="sm" onClick={() => setSelectedDataset("manual")}>Edit paths</Button></div></div> : <><div className="field-stack"><Label htmlFor="survey-source">Survey CSV</Label><Input id="survey-source" value={draft.surveySource} onChange={(event) => setField("surveySource", event.target.value)} placeholder="data/.../info.csv" className="h-10 bg-background font-mono text-xs" /></div><div className="field-stack"><Label htmlFor="survey-schema">Survey schema YAML</Label><Input id="survey-schema" list="survey-schema-options" value={draft.surveySchema} onChange={(event) => setField("surveySchema", event.target.value)} placeholder="config/surveys/...yaml" className="h-10 bg-background font-mono text-xs" /><datalist id="survey-schema-options">{data.surveys.map((item) => <option key={item.path} value={item.path} />)}</datalist></div>{needsTarget && <div className="field-stack sm:col-span-2"><Label htmlFor="target-source">Target aggregate JSON</Label><Input id="target-source" value={draft.target} onChange={(event) => setField("target", event.target.value)} placeholder="data/.../aggregate_results.json" className="h-10 bg-background font-mono text-xs" /></div>}</>}
              {needsTarget && <div className="field-stack"><Label htmlFor="target-subgroup">Target subgroup <span className="font-normal text-muted-foreground">(grouped files only)</span></Label><Input id="target-subgroup" list="target-subgroup-options" value={draft.targetSubgroup} onChange={(event) => setField("targetSubgroup", event.target.value)} placeholder={targetSubgroups?.default ? `${targetSubgroups.default} (file default)` : "Leave empty for a flat target file"} className="h-10 bg-background font-mono text-xs" /><datalist id="target-subgroup-options">{(targetSubgroups?.subgroups || []).map((name) => <option key={name} value={name} />)}</datalist><p className="text-xs leading-5 text-muted-foreground">{targetSubgroups ? `This file holds ${targetSubgroups.subgroups.length} subgroups.` : "A grouped target file holds one distribution per subgroup; preflight lists them."}</p></div>}
              {draft.command === "regress" && <div className="field-stack sm:col-span-2"><Label htmlFor="basis-source">Existing proxy basis directory</Label><Input id="basis-source" value={draft.basis} onChange={(event) => setField("basis", event.target.value)} placeholder="outputs/.../repeat_01" className="h-10 bg-background font-mono text-xs" /></div>}
              {draft.command === "evaluate" && <div className="field-stack sm:col-span-2"><Label htmlFor="predictions-source">Predictions JSON</Label><Input id="predictions-source" value={draft.predictions} onChange={(event) => setField("predictions", event.target.value)} placeholder="outputs/.../predictions.json" className="h-10 bg-background font-mono text-xs" /></div>}
            </div>
          </section>

          {needsGeneration && <section id="builder-step-stage1" className="surface-card builder-step overflow-hidden">
            <div className="border-b border-border/70 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><span className="step-index">{stepNumber("stage1")}</span><h2 className="section-title">Stage 1 · Proxy population</h2></div><p className="section-copy">Generate personas from structured attributes and ask them the survey.</p></div>
            <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
              <><div className="field-stack"><Label>Generation strategy</Label><Select value={draft.generationMode} onValueChange={(value) => setField("generationMode", value as ExperimentDraft["generationMode"])}><SelectTrigger className="h-10 w-full bg-background"><SelectValue /></SelectTrigger><SelectContent>{data.capabilities.generationModes.map((mode) => <SelectItem key={mode.id} value={mode.id}>{mode.label}</SelectItem>)}</SelectContent></Select></div><div className="field-stack"><Label htmlFor="target-n">Target proxy count</Label><Input id="target-n" type="number" min={1} value={draft.targetN} onChange={(event) => setField("targetN", Number(event.target.value))} className="h-10 bg-background" /></div><div className="field-stack sm:col-span-2"><Label>Backend profile</Label><Select value={draft.backendProfile} onValueChange={(value) => setField("backendProfile", value)}><SelectTrigger className="h-10 w-full bg-background"><SelectValue placeholder="Choose a backend" /></SelectTrigger><SelectContent>{data.backends.map((item) => <SelectItem key={item.path} value={item.path}>{item.name} · {item.local ? "local" : "metered"}</SelectItem>)}</SelectContent></Select></div>{advancedOpen && draft.generationMode === "active" && <><div className="field-stack"><Label htmlFor="initial-n">Initial proxies</Label><Input id="initial-n" type="number" min={1} max={draft.targetN} value={draft.initialN} onChange={(event) => setField("initialN", Number(event.target.value))} className="h-10 bg-background" /></div><div className="field-stack"><Label htmlFor="update-steps">Update steps</Label><Input id="update-steps" type="number" min={1} value={draft.updateSteps} onChange={(event) => setField("updateSteps", Number(event.target.value))} className="h-10 bg-background" /></div><div className="field-stack sm:col-span-2"><Label htmlFor="attribute-bank">Attribute bank YAML</Label><Input id="attribute-bank" value={draft.attributeBank} onChange={(event) => setField("attributeBank", event.target.value)} className="h-10 bg-background font-mono text-xs" /></div></>}<PopulationBudgetPreview targetN={draft.targetN} /></>
            </div>
          </section>}

          {needsModel && <section id="builder-step-stage2" className="surface-card builder-step overflow-hidden">
            <div className="border-b border-border/70 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><span className="step-index">{stepNumber("stage2")}</span><h2 className="section-title">Stage 2 · Weighted ensemble</h2></div><p className="section-copy">Weight the proxies so their answers match the target, choosing the penalty on held-out questions.</p></div>
            <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
              <div className="field-stack sm:col-span-2"><Label htmlFor="reference-distribution">Reference distribution <span className="font-normal text-muted-foreground">(optional)</span></Label><Input id="reference-distribution" value={draft.referenceDistribution} onChange={(event) => setField("referenceDistribution", event.target.value)} placeholder="data/.../earlier_wave.json" className="h-10 bg-background font-mono text-xs" /><p className="text-xs leading-5 text-muted-foreground">Re-anchors Stage 2: this distribution enters every prediction with weight one and the agents fit each target's change from it, as when refitting a survey to a later wave.</p></div>
              <Stage2Controls draft={draft} setField={setField} caps={data.capabilities} advancedOpen={advancedOpen} usesSolver={usesSolver} />
            </div>
          </section>}

          {needsTarget && <section id="builder-step-evaluation" className="surface-card builder-step overflow-hidden">
            <div className="border-b border-border/70 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><span className="step-index">{stepNumber("evaluation")}</span><h2 className="section-title">Evaluation</h2></div><p className="section-copy">How held-out predictions are scored against the target.</p></div>
            <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
              {advancedOpen ? <><MetricChooser draft={draft} setField={setField} caps={data.capabilities} /><EvaluationChooser draft={draft} setField={setField} caps={data.capabilities} /></> : <p className="text-sm leading-6 text-muted-foreground sm:col-span-2">Scored on the test split with {draft.metrics.length} {draft.metrics.length === 1 ? "metric" : "metrics"}; distributions: {data.capabilities.evaluation.distributionPolicies.find((item) => item.id === draft.distributionPolicy)?.label.replace(" (default)", "").toLowerCase() || draft.distributionPolicy}. Open advanced settings to change the metrics or how predictions are scored.</p>}
            </div>
          </section>}

          <section id="builder-step-review" className="surface-card builder-step overflow-hidden">
            <div className="border-b border-border/70 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><span className="step-index">{stepNumber("review")}</span><h2 className="section-title">Review the manifest</h2></div><p className="section-copy">This preview updates immediately as the form changes. Saved paths are normalized relative to the manifest.</p></div>
            <pre className="max-h-[520px] overflow-auto reading-surface p-5">{yamlPreview}</pre>
          </section>

          {activeJob && (
            <section className="surface-card overflow-hidden">
              <div className="flex items-center justify-between border-b px-5 py-4 sm:px-6"><div><h2 className="section-title">Process log</h2><p className="section-copy">Live stdout and stderr from the local P2P process.</p></div><Badge variant="outline" className={activeJob.status === "failed" ? "text-destructive" : ""}>{activeJob.status}</Badge></div>
              <pre className="max-h-[340px] min-h-40 overflow-auto reading-surface p-5">{activeJob.log.length ? activeJob.log.join("\n") : "Process started. Waiting for output…"}</pre>
            </section>
          )}
        </div>

        <aside className="min-w-0 space-y-5 xl:sticky xl:top-24 xl:self-start">
          <section className="run-card run-card-instrument">
            <div className="mb-5 flex items-center justify-between"><div><p className="text-sm font-semibold">{activeJob ? `Run ${activeJob.status}` : ready ? "Ready to create" : "Complete required fields"}</p><p className="mt-0.5 text-xs text-white/55">{activeJob ? activeJob.id : createdManifest?.path || "No data leaves this machine"}</p></div><span className={`size-3 rounded-full ${activeJob?.status === "failed" ? "bg-rose-400" : activeJob?.status === "paused" ? "bg-amber-400" : ready ? "bg-teal-400" : "bg-white/30"}`} /></div>
            <div className="space-y-4 rounded-xl border border-white/10 bg-black/15 p-4 text-sm"><div className="flex justify-between gap-4"><span className="text-white/60">Workflow</span><span className="font-mono">{draft.command}</span></div>{needsGeneration && <div className="flex justify-between gap-4"><span className="text-white/60">Strategy</span><span>{draft.generationMode}</span></div>}<div className="flex justify-between gap-4"><span className="text-white/60">Model</span><span>{needsModel ? selectedModel?.label || draft.modelMethod : "—"}</span></div>{fitsPenalty && <div className="flex justify-between gap-4"><span className="text-white/60">Configuration</span><span className="max-w-[220px] truncate text-right">{selectedConfiguration?.label || draft.configuration}</span></div>}{reanchored && <div className="flex justify-between gap-4"><span className="text-white/60">Re-anchored to</span><span className="max-w-[220px] truncate font-mono text-xs" title={draft.referenceDistribution}>{draft.referenceDistribution.split("/").at(-1)}</span></div>}{reanchored && !referenceReady && <p className="text-xs text-rose-300">Choose a fitted method and a configuration without a reference respondent or intercept.</p>}{usesSolver && needsModel && <div className="flex justify-between gap-4"><span className="text-white/60">Solver</span><span className="font-mono text-xs">{draft.solver || `${data.capabilities.stage2.defaultSolver} (default)`}</span></div>}{needsGeneration && <div className="flex justify-between gap-4"><span className="text-white/60">Backend</span><span className="max-w-[220px] truncate">{backendLabel}</span></div>}<div className="flex justify-between gap-4"><span className="text-white/60">Output</span><span className="max-w-[220px] truncate font-mono text-xs">{output}</span></div><div className="flex justify-between gap-4"><span className="text-white/60">Network</span><span>{needsGeneration && selectedBackend && !selectedBackend.local ? "Provider API" : "Local only"}</span></div></div>
            {!activeJob || ["complete", "failed", "stopped", "interrupted"].includes(activeJob.status) ? preflight?.valid ? <Button size="lg" className="brand-run-button mt-5 h-11 w-full bg-white hover:bg-white/90" onClick={() => void start()} disabled={starting}>{starting ? <LoaderCircle className="animate-spin" /> : <Play className="fill-current" />} Create YAML &amp; run</Button> : <Button size="lg" className="brand-run-button mt-5 h-11 w-full bg-white hover:bg-white/90" onClick={() => void reviewPreflight()} disabled={starting || !ready}>{starting ? <LoaderCircle className="animate-spin" /> : <ShieldCheck />} Review preflight</Button> : activeJob.status === "running" ? <div className="mt-5 grid grid-cols-[1fr_auto] gap-2">{data.capabilities.processControls.pause && <Button size="lg" className="brand-run-button h-11 bg-white hover:bg-white/90" onClick={() => void act("pause")}><Pause className="fill-current" /> Pause</Button>}<Button size={data.capabilities.processControls.pause ? "icon-lg" : "lg"} variant="outline" className="h-11 border-white/20 bg-white/5 text-white hover:bg-white/10 hover:text-white" onClick={() => void act("stop")}><Square className="fill-current" />{!data.capabilities.processControls.pause && " Stop"}</Button></div> : <div className="mt-5 grid grid-cols-2 gap-2">{data.capabilities.processControls.resume && <Button size="lg" className="brand-run-button h-11 bg-white hover:bg-white/90" onClick={() => void act("resume")}><Play className="fill-current" /> Resume</Button>}<Button size="lg" variant="outline" className="h-11 border-white/20 bg-white/5 text-white hover:bg-white/10 hover:text-white" onClick={() => void act("stop")}><Square className="fill-current" /> Stop</Button></div>}
          </section>
          {preflight && <section className={`surface-card border-l-4 p-5 ${preflight.valid ? "border-l-emerald-500" : "border-l-rose-500"}`}><div className="flex items-center justify-between gap-3"><h3 className="text-sm font-semibold">{preflight.valid ? "Preflight passed" : "Preflight blocked"}</h3><Badge variant="outline">{preflight.paid ? "Cost unknown" : "Offline / local"}</Badge></div><dl className="mt-4 space-y-3 text-xs"><div><dt className="text-muted-foreground">Config digest</dt><dd className="mt-1 truncate font-mono" title={preflight.configDigest}>{preflight.configDigest}</dd></div><div><dt className="text-muted-foreground">Output</dt><dd className="mt-1 truncate font-mono">{preflight.output}</dd></div></dl>{preflight.warnings.length > 0 && <ul className="mt-4 space-y-2 text-xs leading-5 text-amber-700">{preflight.warnings.map((warning) => <li key={warning} className="flex gap-2"><TriangleAlert className="mt-0.5 size-3.5 shrink-0" />{warning}</li>)}</ul>}</section>}
          <section className="surface-card p-5"><div className="mb-4 flex items-center gap-2"><TerminalSquare className="size-4 text-muted-foreground" /><h3 className="text-sm font-semibold">CLI equivalent</h3></div><pre className="overflow-x-auto rounded-lg bg-muted p-3 font-mono text-[11px] leading-5">p2p {draft.command} --config {manifestPath} --output {output}{solverFlag}</pre></section>
        </aside>
      </div>
    </>
  );
}

// Live and recent runs: every local process from launch to result, shown at the
// top of the Runs page. Records survive Studio restarts; process controls apply
// only to attached live jobs. The raw process log sits behind a disclosure.
function JobsView({ data, onRefresh, onRun, onDuplicate }: { data: Bootstrap; onRefresh: () => Promise<void>; onRun: (path: string) => void; onDuplicate: (manifest: string) => void }) {
  const [jobs, setJobs] = useState<Job[]>(data.jobs);
  const [selectedId, setSelectedId] = useState("");
  const [showAll, setShowAll] = useState(false);
  const selected = jobs.find((job) => job.id === selectedId) || jobs[0];
  const activeCount = jobs.filter((job) => ["running", "paused"].includes(job.status)).length;
  const failedCount = jobs.filter((job) => ["failed", "interrupted"].includes(job.status)).length;

  const refreshJobs = useCallback(async () => {
    try {
      const payload = await api<{ jobs: Job[] }>("/api/jobs");
      const ordered = [...payload.jobs].sort((left, right) => right.createdAt.localeCompare(left.createdAt));
      setJobs(ordered);
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not refresh jobs."); }
  }, [selectedId]);

  useEffect(() => {
    void refreshJobs();
    if (!jobs.some((job) => ["running", "paused"].includes(job.status))) return;
    const timer = window.setInterval(() => void refreshJobs(), 1500);
    return () => window.clearInterval(timer);
  }, [jobs.some((job) => ["running", "paused"].includes(job.status)), refreshJobs]);

  async function act(job: Job, action: "pause" | "resume" | "stop") {
    try {
      await api<Job>(`/api/jobs/${job.id}/action`, { method: "POST", body: JSON.stringify({ action }) });
      toast.success(action === "pause" ? "Job paused." : action === "resume" ? "Job resumed." : "Job stopped.");
      await refreshJobs();
      if (action === "stop") await onRefresh();
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not update this job."); }
  }

  async function openResult(job: Job) {
    await onRefresh();
    onRun(job.output);
  }

  if (!jobs.length) return null;
  const listed = showAll ? jobs : jobs.slice(0, 4);
  return <section className="mb-6" aria-labelledby="live-runs-title">
    <div className="mb-3 flex items-end justify-between gap-4"><div><h2 id="live-runs-title" className="section-title">Live and recent runs</h2><p className="section-copy">{activeCount ? `${activeCount} running · ` : ""}{failedCount ? `${failedCount} need attention · ` : ""}{jobs.length} started from Studio</p></div><Button variant="ghost" size="sm" onClick={() => void refreshJobs()}><RefreshCw /> Refresh</Button></div>
    <div className={`grid gap-5 ${selected && selectedId ? "xl:grid-cols-[minmax(320px,.8fr)_minmax(0,1.2fr)]" : ""}`}>
      <section className="surface-card self-start overflow-hidden">{listed.map((job) => <button key={job.id} type="button" aria-expanded={selectedId === job.id} onClick={() => setSelectedId((current) => current === job.id ? "" : job.id)} className={`flex w-full items-start gap-3 border-b px-5 py-4 text-left last:border-0 ${selectedId === job.id ? "bg-primary/7" : "hover:bg-muted/45"}`}><span className={`mt-1 size-2.5 shrink-0 rounded-full ${job.status === "complete" ? "bg-emerald-500" : job.status === "running" ? "bg-sky-500" : job.status === "paused" ? "bg-amber-500" : "bg-rose-500"}`} /><span className="min-w-0 flex-1"><strong className="block truncate text-sm">{job.output}</strong><span className="mt-1 block truncate font-mono text-[11px] text-muted-foreground">{job.command} · {job.id}</span><span className="mt-1 block text-xs text-muted-foreground">{formatDate(job.createdAt)}</span></span><Badge variant="outline" className="capitalize">{job.status}</Badge></button>)}{jobs.length > 4 && <button type="button" className="w-full px-5 py-2.5 text-left text-xs text-muted-foreground hover:bg-muted/45" onClick={() => setShowAll((value) => !value)}>{showAll ? "Show fewer" : `Show all ${jobs.length}`}</button>}</section>
      {selectedId && selected ? <section className="surface-card min-w-0 overflow-hidden"><div className="flex flex-col gap-3 border-b px-5 py-4 sm:flex-row sm:items-start sm:justify-between"><div className="min-w-0"><div className="flex items-center gap-2"><h2 className="truncate text-base font-semibold">{selected.output}</h2><Badge variant="outline" className="capitalize">{selected.status}</Badge></div><p className="mt-1 truncate font-mono text-xs text-muted-foreground">{selected.config}</p></div><div className="flex shrink-0 flex-wrap gap-2"><Button variant="outline" size="sm" onClick={() => onDuplicate(selected.config)}><FileCode2 /> Duplicate setup</Button>{selected.status === "running" && data.capabilities.processControls.pause && <Button variant="outline" size="sm" onClick={() => void act(selected, "pause")}><Pause /> Pause</Button>}{selected.status === "paused" && data.capabilities.processControls.resume && <Button variant="outline" size="sm" onClick={() => void act(selected, "resume")}><Play /> Resume</Button>}{["running", "paused"].includes(selected.status) && <Button variant="outline" size="sm" onClick={() => void act(selected, "stop")}><Square /> Stop</Button>}{selected.status === "complete" && <Button size="sm" onClick={() => void openResult(selected)}><BarChart3 /> Open result</Button>}</div></div><dl className="grid gap-3 border-b bg-muted/20 p-5 text-sm sm:grid-cols-3"><div><dt className="text-xs text-muted-foreground">Command</dt><dd className="mt-1 font-mono">{selected.command}</dd></div><div><dt className="text-xs text-muted-foreground">Recovery</dt><dd className="mt-1 capitalize">{selected.recovery.replaceAll("-", " ")}</dd></div><div><dt className="text-xs text-muted-foreground">Process</dt><dd className="mt-1 font-mono">{["running", "paused"].includes(selected.status) && selected.pid ? `PID ${selected.pid}` : selected.status === "complete" ? "Finished" : "Detached"}</dd></div></dl><details className="job-log" open={["failed", "interrupted"].includes(selected.status) || undefined}><summary className="flex cursor-pointer items-center gap-2 px-5 py-3 text-sm font-semibold"><TerminalSquare className="size-4 text-muted-foreground" /> Process log <span className="font-normal text-muted-foreground">· {selected.log.length} lines</span></summary><pre className="max-h-[560px] min-h-40 overflow-auto reading-surface p-5">{selected.log.length ? selected.log.join("\n") : "No process output was captured."}</pre></details></section> : null}
    </div>
  </section>;
}

function BudgetView({ data }: { data: Bootstrap }) {
  const [selectedPath, setSelectedPath] = useState(data.budgets[0]?.path || "");
  const [detail, setDetail] = useState<BudgetDetail | null>(null);
  const summary = data.budgets.find((item) => item.path === selectedPath);
  useEffect(() => { if (!selectedPath) return; setDetail(null); void api<BudgetDetail>(`/api/budget?path=${encodeURIComponent(selectedPath)}`).then(setDetail).catch((error) => toast.error(error.message)); }, [selectedPath]);
  const scenarios = detail?.scenarios || [];
  const maxCost = Math.max(...scenarios.map((item) => item.total_cost_usd.high || 0), 0.0001);
  return (
    <>
      <PageHeading eyebrow="Workspace / Budgets" title="Forecast cost before execution" copy="The existing budget forecast artifacts are now part of the Studio: compare models, inspect uncertainty, and see module-level cost drivers." action={data.budgets.length ? <Select value={selectedPath} onValueChange={setSelectedPath}><SelectTrigger className="w-full bg-card sm:w-[360px]"><SelectValue /></SelectTrigger><SelectContent>{data.budgets.map((item) => <SelectItem key={item.path} value={item.path}>{item.path}</SelectItem>)}</SelectContent></Select> : undefined} />
      {!summary ? <EmptyState icon={CircleDollarSign} title="No budget forecasts yet" copy="Run `p2p budget` and the generated budget.json will appear here." /> : <div className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-3">{[["Confidence", `${Math.round((summary.confidence || 0) * 100)}%`], ["Runs learned from", formatNumber(summary.historicalRuns)], ["Pilot records", formatNumber(summary.pilotRecords)]].map(([label, value]) => <div key={label} className="surface-card p-5"><p className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">{label}</p><p className="mt-2 text-2xl font-semibold">{value}</p></div>)}</div>
        <section className="surface-card p-5 sm:p-6"><div className="mb-6"><h2 className="section-title">Scenario comparison</h2><p className="section-copy">Point estimate with low-to-high forecast interval.</p></div><div className="space-y-6">{scenarios.map((scenario) => { const high = scenario.total_cost_usd.high || 0; const point = scenario.total_cost_usd.point || 0; return <div key={scenario.name} className="grid gap-3 md:grid-cols-[190px_minmax(240px,1fr)_130px_150px] md:items-center"><div><p className="font-semibold">{scenario.name}</p><p className="text-xs text-muted-foreground">{scenario.modules?.[0]?.model_name || "configured model"}</p></div><div className="relative h-9 rounded-lg bg-muted"><div className="absolute inset-y-2 left-0 rounded-md bg-primary/20" style={{ width: `${high / maxCost * 100}%` }} /><div className="absolute inset-y-1.5 w-1 rounded-full bg-primary" style={{ left: `calc(${point / maxCost * 100}% - 2px)` }} /><span className="absolute inset-0 flex items-center px-3 font-mono text-xs">{formatMoney(scenario.total_cost_usd.low)} – {formatMoney(high)}</span></div><div className="text-right"><p className="font-mono font-semibold">{formatMoney(point)}</p><p className="text-xs text-muted-foreground">forecast cost</p></div><div className="text-right"><p className="font-mono font-semibold">{formatNumber(scenario.total_tokens.point)} tokens</p><p className="text-xs text-muted-foreground">{scenario.budget_status || "No ceiling"}</p></div></div>;})}</div></section>
        <div className="grid gap-5 xl:grid-cols-[1.5fr_1fr]">
          <section className="surface-card overflow-hidden"><div className="border-b px-5 py-4 sm:px-6"><h2 className="section-title">Module breakdown</h2><p className="section-copy">Token and cost point estimates for each workflow module.</p></div><div className="overflow-x-auto"><table className="w-full min-w-[650px] text-sm"><thead className="bg-muted/40 text-left text-xs uppercase tracking-[.06em] text-muted-foreground"><tr><th className="px-5 py-3">Scenario / module</th><th className="px-5 py-3">Model</th><th className="px-5 py-3 text-right">Input</th><th className="px-5 py-3 text-right">Output</th><th className="px-5 py-3 text-right">Cost</th></tr></thead><tbody>{scenarios.flatMap((scenario) => (scenario.modules || []).map((module) => <tr key={`${scenario.name}-${module.module_name}`} className="border-t"><td className="px-5 py-3"><span className="block font-medium">{module.module_name}</span><span className="text-xs text-muted-foreground">{scenario.name}</span></td><td className="px-5 py-3">{module.model_name}</td><td className="px-5 py-3 text-right font-mono">{formatNumber(module.input_tokens.point)}</td><td className="px-5 py-3 text-right font-mono">{formatNumber(module.output_tokens.point)}</td><td className="px-5 py-3 text-right font-mono">{formatMoney(module.cost_usd.point)}</td></tr>))}</tbody></table></div></section>
          <section className="surface-card p-5 sm:p-6"><div className="mb-4 flex items-center gap-2"><TriangleAlert className="size-4 text-amber-500" /><h2 className="section-title">Forecast notes</h2></div><ul className="space-y-3 text-sm leading-6 text-muted-foreground">{(detail?.notes || []).map((note) => <li key={note} className="border-b pb-3 last:border-0">{note}</li>)}</ul></section>
        </div>
      </div>}
    </>
  );
}

function DataSources({ data, onRefresh }: { data: Bootstrap; onRefresh: () => Promise<void> }) {
  const [name, setName] = useState("My survey");
  const [source, setSource] = useState("examples/studio_demo/survey.csv");
  const [schema, setSchema] = useState("examples/studio_demo/survey.yaml");
  const [target, setTarget] = useState("examples/studio_demo/targets.json");
  const [inspection, setInspection] = useState<DatasetInspection | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);

  async function inspect() {
    setBusy(true);
    try {
      const result = await api<DatasetInspection>("/api/datasets/inspect", { method: "POST", body: JSON.stringify({ source, schema: schema || undefined, target: target || undefined, mapping: schema ? undefined : mapping }) });
      setInspection(result);
      if (!schema) setMapping((current) => ({ ...current, ...Object.fromEntries(Object.entries(result.mapping).filter((entry): entry is [string, string] => typeof entry[1] === "string")) }));
      result.valid ? toast.success("Dataset contract is valid.") : toast.error("Dataset inspection found blocking issues.");
    } catch (error) {
      setInspection(null);
      toast.error(error instanceof Error ? error.message : "Could not inspect this dataset.");
    } finally { setBusy(false); }
  }

  async function register() {
    if (!inspection?.valid) return;
    setBusy(true);
    try {
      await api<Dataset>("/api/datasets", { method: "POST", body: JSON.stringify({ name, source, schema: schema || undefined, target: target || undefined, mapping: schema ? undefined : mapping }) });
      toast.success("Dataset registered locally with source digests.");
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not register this dataset.");
    } finally { setBusy(false); }
  }

  return (
    <>
      <PageHeading eyebrow="Workspace / Data" title="Inspect and register local survey data" copy="Map the survey contract, preview option support, verify train/validation/test splits, and compare aggregate target coverage before saving a descriptor. Studio only reads paths inside this repository." />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.35fr)_360px]">
        <div className="space-y-5">
          <section className="surface-card overflow-hidden">
            <div className="border-b px-5 py-4 sm:px-6"><h2 className="section-title">Data inspection</h2><p className="section-copy">Start with the bundled synthetic example or enter repository-relative paths to your own files.</p></div>
            <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
              <div className="field-stack"><Label htmlFor="dataset-name">Dataset name</Label><Input id="dataset-name" value={name} onChange={(event) => setName(event.target.value)} /></div>
              <div className="field-stack"><Label htmlFor="dataset-source">Survey CSV</Label><Input id="dataset-source" className="font-mono text-xs" value={source} onChange={(event) => { setSource(event.target.value); setInspection(null); setMapping({}); }} /></div>
              <div className="field-stack"><Label htmlFor="dataset-schema">Schema YAML <span className="font-normal text-muted-foreground">(optional)</span></Label><Input id="dataset-schema" className="font-mono text-xs" value={schema} placeholder="Leave blank to map CSV columns here" onChange={(event) => { setSchema(event.target.value); setInspection(null); }} /></div>
              <div className="field-stack"><Label htmlFor="dataset-target">Aggregate targets JSON</Label><Input id="dataset-target" className="font-mono text-xs" value={target} onChange={(event) => { setTarget(event.target.value); setInspection(null); }} /></div>
              <div className="flex flex-wrap gap-2 sm:col-span-2"><Button onClick={() => void inspect()} disabled={busy || !source}>{busy ? <LoaderCircle className="animate-spin" /> : <Search />} Inspect data</Button>{inspection?.valid && <Button variant="outline" onClick={() => void register()} disabled={busy || !name.trim()}><Database /> Register dataset</Button>}</div>
            </div>
          </section>

          {inspection && <section className="surface-card overflow-hidden"><div className="flex items-start justify-between gap-4 border-b px-5 py-4 sm:px-6"><div><h2 className="section-title">Inspection report</h2><p className="section-copy">{inspection.questionCount} questions · {inspection.optionRowCount} option rows · {inspection.targetCoverage.matched} targets matched</p></div><Badge variant="outline" className={inspection.valid ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-rose-200 bg-rose-50 text-rose-700"}>{inspection.valid ? "Ready" : "Blocked"}</Badge></div>
            {!schema && <div className="border-b bg-muted/20 p-5 sm:p-6"><div className="mb-4"><h3 className="text-sm font-semibold">Map CSV columns</h3><p className="section-copy">Studio inferred what it could. Confirm each required survey field; registration will save this mapping as an explicit schema YAML.</p></div><div className="grid gap-4 sm:grid-cols-2">{[["question_id", "Question ID"], ["question_text", "Question text"], ["code_to_answer", "Option mapping"], ["split", "Train/valid/test split"]].map(([key, label]) => <div key={key} className="field-stack"><Label>{label}</Label><select value={mapping[key] || ""} onChange={(event) => setMapping((current) => ({ ...current, [key]: event.target.value }))} className="h-10 rounded-md border bg-background px-3 text-sm"><option value="">Choose a column…</option>{inspection.headers.map((header) => <option key={header} value={header}>{header}</option>)}</select></div>)}</div><Button variant="outline" className="mt-4" onClick={() => void inspect()} disabled={busy || Object.values(mapping).filter(Boolean).length < 4}>{busy ? <LoaderCircle className="animate-spin" /> : <ShieldCheck />} Apply mapping &amp; recheck</Button></div>}
            <div className="grid gap-3 border-b p-5 sm:grid-cols-3">{["train", "valid", "test"].map((split) => <div key={split} className={`rounded-xl border p-4 split-card-${split}`}><p className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">{split === "valid" ? "Validation" : split}</p><p className="mt-1 font-mono text-2xl font-semibold">{inspection.splitCounts[split] || 0}</p><p className="text-xs text-muted-foreground">questions</p></div>)}</div>
            {(inspection.errors.length > 0 || inspection.warnings.length > 0) && <div className="grid gap-4 border-b p-5 sm:grid-cols-2">{inspection.errors.length > 0 && <div><p className="mb-2 text-xs font-semibold uppercase tracking-[.08em] text-rose-700">Blocking issues</p><ul className="space-y-2 text-sm leading-5 text-rose-700">{inspection.errors.map((item) => <li key={item}>{item}</li>)}</ul></div>}{inspection.warnings.length > 0 && <div><p className="mb-2 text-xs font-semibold uppercase tracking-[.08em] text-amber-700">Warnings</p><ul className="space-y-2 text-sm leading-5 text-amber-700">{inspection.warnings.map((item) => <li key={item}>{item}</li>)}</ul></div>}</div>}
            <div className="overflow-x-auto"><table className="w-full min-w-[700px] text-sm"><thead className="bg-muted/40 text-left text-xs uppercase tracking-[.06em] text-muted-foreground"><tr><th className="px-5 py-3">Question ID</th><th className="px-5 py-3">Question</th><th className="px-5 py-3">Split</th><th className="px-5 py-3">Option support</th></tr></thead><tbody>{inspection.preview.map((row) => <tr key={row.qid} className="border-t"><td className="px-5 py-3 font-mono text-xs">{row.qid || "—"}</td><td className="max-w-md px-5 py-3">{row.question || "—"}</td><td className="px-5 py-3"><Badge variant="outline">{row.split}</Badge></td><td className="px-5 py-3 text-muted-foreground">{Object.values(row.options).join(" · ") || "Unparseable"}</td></tr>)}</tbody></table></div></section>}
        </div>

        <aside className="space-y-5">
          <section className="surface-card overflow-hidden"><div className="border-b px-5 py-4"><h2 className="section-title">Registered datasets</h2><p className="section-copy">Portable descriptors with pinned source digests.</p></div>{data.datasets.length ? data.datasets.map((dataset) => <div key={dataset.path} className="border-b p-5 last:border-0"><div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="truncate font-semibold">{dataset.name}</p><p className="truncate font-mono text-[11px] text-muted-foreground">{dataset.path}</p></div><Badge variant="secondary">{dataset.questionCount ?? "—"} Q</Badge></div><div className="mt-3 flex flex-wrap gap-1.5">{Object.entries(dataset.splitCounts || {}).map(([split, count]) => <Badge key={split} variant="outline">{split} {count}</Badge>)}</div></div>) : <div className="p-5 text-sm leading-6 text-muted-foreground">No datasets have been registered yet. Inspection does not alter source files.</div>}</section>
          <section className="surface-card overflow-hidden"><div className="border-b px-5 py-4"><h2 className="section-title">Available schemas</h2><p className="section-copy">Existing column mappings under config/.</p></div>{data.surveys.slice(0, 12).map((survey) => <div key={survey.path} className="border-b p-4 last:border-0"><p className="truncate text-sm font-semibold">{survey.name}</p><p className="truncate font-mono text-[11px] text-muted-foreground">{survey.path}</p></div>)}</section>
        </aside>
      </div>
    </>
  );
}

function Backends({ data }: { data: Bootstrap }) {
  const [checks, setChecks] = useState<Record<string, BackendCheck>>({});
  const [checking, setChecking] = useState("");
  const adapterNames: Record<string, string> = {
    openai: "OpenAI",
    gemini: "Google Gemini",
    deepseek: "DeepSeek",
    ollama: "Ollama",
    dummy: "Dummy",
  };
  const adapterOrder = ["openai", "gemini", "deepseek", "ollama", "dummy"];
  const adapters = data.capabilities.backends
    .map((capability) => ({
      capability,
      profiles: data.backends.filter((profile) => profile.type === capability.id),
    }))
    .sort((left, right) => {
      const leftIndex = adapterOrder.indexOf(left.capability.id);
      const rightIndex = adapterOrder.indexOf(right.capability.id);
      return (leftIndex < 0 ? adapterOrder.length : leftIndex) - (rightIndex < 0 ? adapterOrder.length : rightIndex)
        || left.capability.id.localeCompare(right.capability.id);
    });

  async function check(path: string) {
    setChecking(path);
    try {
      const result = await api<BackendCheck>("/api/backends/check", { method: "POST", body: JSON.stringify({ path }) });
      setChecks((current) => ({ ...current, [path]: result }));
      result.ready ? toast.success(`${result.name} is ready.`) : toast.error(`${result.name} needs attention.`);
    } catch (error) { toast.error(error instanceof Error ? error.message : "Backend check failed."); }
    finally { setChecking(""); }
  }
  return (
    <>
      <PageHeading eyebrow="Workspace / Backends" title="Model backends" copy="Review every backend adapter available to P2P, then see which profiles, dependencies, and credentials make it runnable. Checks stay local and never send a provider request." />
      <div className="backend-card-grid">
        {adapters.map(({ capability, profiles }) => {
          const results = profiles.map((profile) => checks[profile.path]).filter(Boolean);
          const failed = results.some((result) => !result.ready);
          const verified = profiles.length > 0 && results.length === profiles.length && results.every((result) => result.ready);
          const status = failed
            ? "Needs attention"
            : verified
              ? "Verified"
              : profiles.length === 0
                ? "Profile required"
                : !capability.dependencyAvailable
                  ? "Dependency missing"
                  : !capability.local && !capability.credentialAvailable
                    ? "Credential missing"
                    : capability.local
                      ? "Local"
                      : "Configured";
          const statusClass = failed || !capability.dependencyAvailable
            ? "border-rose-200 bg-rose-50 text-rose-700"
            : verified || (profiles.length > 0 && (capability.local || capability.credentialAvailable))
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-amber-200 bg-amber-50 text-amber-700";

          return (
            <section key={capability.id} className="backend-card surface-card">
              <div className="backend-identity">
                <div className="flex items-start justify-between gap-4">
                  <span className="backend-adapter-icon"><Server className="size-5" /></span>
                  <Badge variant="outline" className={statusClass}>{status}</Badge>
                </div>
                <h2>{adapterNames[capability.id] || capability.id}</h2>
                <p>{capability.local ? "Local runtime adapter" : "Metered provider adapter"} · <code>{capability.id}</code></p>
              </div>
              <dl className="backend-card-requirements">
                <div><dt>Dependency</dt><dd>{capability.dependency || "Built in"} · {capability.dependencyAvailable ? "ready" : "missing"}</dd></div>
                <div><dt>Credential</dt><dd>{capability.local ? "Not required" : capability.credentialName || "Not declared"}</dd></div>
                <div><dt>Model ID</dt><dd>{capability.modelRequired ? "Required in every profile" : "Has a default"}</dd></div>
              </dl>
              {!!capability.options?.length && <div className="border-t px-5 py-3"><p className="mb-2 text-[11px] font-semibold uppercase tracking-[.075em] text-muted-foreground">Settings (profile kwargs)</p><div className="flex flex-wrap gap-1.5">{capability.options.map((option) => <code key={option.name} className="rounded bg-muted px-1.5 py-0.5 text-[11px]" title={option.required ? "Required" : `Default: ${JSON.stringify(option.default)}`}>{option.name}{option.default !== null && option.default !== undefined ? `=${typeof option.default === "string" ? option.default : JSON.stringify(option.default)}` : ""}</code>)}</div></div>}

              <div className="backend-card-profiles">
                {profiles.length === 0 ? (
                  <div className="backend-card-empty">
                    <p className="font-medium text-foreground">No profile configured</p>
                    <p className="mt-1">Add <code className="rounded bg-muted px-1 py-0.5">config/backends/{capability.id}.yaml</code> to select this adapter in an experiment.</p>
                  </div>
                ) : profiles.map((profile) => {
                  const result = checks[profile.path];
                  return (
                    <div key={profile.path} className="backend-card-profile">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0"><p className="truncate text-sm font-semibold">{profile.name}</p><p className="truncate font-mono text-[11px] text-muted-foreground">{profile.path}</p></div>
                        <Button variant="outline" size="sm" onClick={() => void check(profile.path)} disabled={checking === profile.path}>{checking === profile.path ? <LoaderCircle className="animate-spin" /> : <ShieldCheck />} Check</Button>
                      </div>
                      <div className="mt-3 space-y-2">{profile.modules.map((module) => <div key={module.module} className="flex items-center justify-between gap-3 text-xs"><span className="text-muted-foreground">{module.module.replaceAll("_", " ")}</span><span className="max-w-[60%] truncate font-medium">{module.model}</span></div>)}</div>
                      {result && <div className="mt-3 space-y-2 border-t pt-3">{result.checks.map((item) => <div key={item.label} className="flex gap-2 text-xs"><span className={`mt-1 size-2 shrink-0 rounded-full ${item.ok ? "bg-emerald-500" : "bg-rose-500"}`} /><span><strong>{item.label}</strong><span className="ml-1 text-muted-foreground">{item.detail}</span></span></div>)}<p className="pt-1 text-[11px] text-muted-foreground">No provider request was made.</p></div>}
                    </div>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </>
  );
}

export default function Home() {
  const [view, setView] = useState<View>(() => readRoute().view);
  const [data, setData] = useState<Bootstrap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [docsOpen, setDocsOpen] = useState(false);
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  const [welcomeChecked, setWelcomeChecked] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [focusRunPath, setFocusRunPath] = useState<string | null>(() => readRoute().run);
  const [focusReportPath, setFocusReportPath] = useState<string | null>(() => readRoute().report);
  const [focusManifestPath, setFocusManifestPath] = useState<string | null>(() => readRoute().manifest);
  const [focusComparePaths, setFocusComparePaths] = useState<string[]>(() => readRoute().compare);

  const navigate = useCallback((nextView: View, selection?: { run?: string; report?: string; manifest?: string; compare?: string[] }, replace = false) => {
    const params = new URLSearchParams();
    if (nextView !== "home") params.set("view", nextView);
    if (selection?.run) params.set("run", selection.run);
    if (selection?.report) params.set("report", selection.report);
    if (selection?.manifest) params.set("from", selection.manifest);
    selection?.compare?.forEach((path) => params.append("compare", path));
    const url = `${window.location.pathname}${params.size ? `?${params.toString()}` : ""}`;
    window.history[replace ? "replaceState" : "pushState"]({}, "", url);
    setView(nextView);
    setFocusRunPath(selection?.run || null);
    setFocusReportPath(selection?.report || null);
    setFocusManifestPath(selection?.manifest || null);
    setFocusComparePaths(selection?.compare || []);
  }, []);
  const reflectLibrarySelection = useCallback((selection: { run?: string; report?: string }) => {
    const params = new URLSearchParams({ view: "runs" });
    if (selection.run) params.set("run", selection.run);
    if (selection.report) params.set("report", selection.report);
    window.history.pushState({}, "", `${window.location.pathname}?${params.toString()}`);
    setView("runs");
    setFocusRunPath(null);
    setFocusReportPath(null);
    setFocusManifestPath(null);
    setFocusComparePaths([]);
  }, []);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try { setData(await api<Bootstrap>("/api/bootstrap")); setError(null); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Could not connect to the local P2P server."); }
    finally { setRefreshing(false); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    const handlePopState = () => {
      const route = readRoute();
      setView(route.view);
      setFocusRunPath(route.run);
      setFocusReportPath(route.report);
      setFocusManifestPath(route.manifest);
      setFocusComparePaths(route.compare);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);
  useEffect(() => {
    if (!data || welcomeChecked) return;
    setWelcomeOpen(window.localStorage.getItem(welcomePreferenceKey) !== "complete");
    setWelcomeChecked(true);
  }, [data, welcomeChecked]);

  const finishWelcome = useCallback(() => {
    window.localStorage.setItem(welcomePreferenceKey, "complete");
    setWelcomeOpen(false);
  }, []);

  const runOfflineDemo = useCallback(async () => {
    try {
      const started = await api<Job>("/api/demo", { method: "POST", body: "{}" });
      toast.info("Running the full pipeline offline: simulated residents (rules, not a language model) are generated and surveyed, then Stage 2 is fitted…");
      for (let attempt = 0; attempt < 480; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 250));
        const payload = await api<{ jobs: Job[] }>("/api/jobs");
        const job = payload.jobs.find((item) => item.id === started.id);
        if (!job) throw new Error("The offline demo job record disappeared.");
        if (["failed", "stopped", "interrupted"].includes(job.status)) throw new Error(job.log.at(-1) || `Offline demo ${job.status}.`);
        if (job.status === "complete") {
          await refresh();
          navigate("runs", { run: job.output });
          toast.success("Demo complete. Opening the run: generated population, fitted ensemble and held-out evidence.");
          return;
        }
      }
      throw new Error("The offline demo is still running; open Runs to follow it.");
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "Could not start the offline demo.");
    }
  }, [navigate, refresh]);

  return (
    <main className="p2p-app min-h-screen text-foreground">
      <Toaster position="bottom-right" theme="dark" toastOptions={{ className: "p2p-toast" }} />
      <AppHeader view={view} onNavigate={navigate} onDocs={() => setDocsOpen(true)} onRefresh={() => void refresh()} refreshing={refreshing} />
      <div className="w-full">
        <section key={view} className="workspace-canvas studio-view-enter min-w-0 px-4 py-6 pb-24 sm:px-6 xl:px-10 xl:py-8">
          {error ? <EmptyState icon={TriangleAlert} title="Local server unavailable" copy={`${error} Start the interface with “p2p ui” so the browser can read repository artifacts.`} /> : !data ? <WorkspaceLoading /> : view === "home" ? <WorkspaceHome data={data} onExperiment={() => navigate("new")} onRuns={() => navigate("runs")} onRun={(path) => navigate("runs", { run: path })} onReport={(path) => navigate("runs", { report: path })} onJobs={() => navigate("runs")} onData={() => navigate("data")} onDemo={() => void runOfflineDemo()} /> : view === "new" ? <NewExperiment data={data} onRefresh={refresh} sourceManifestPath={focusManifestPath} /> : view === "runs" ? <ResultsView liveRuns={<JobsView data={data} onRefresh={refresh} onRun={(path) => navigate("runs", { run: path })} onDuplicate={(manifest) => navigate("new", { manifest })} />} data={data} focusRunPath={focusRunPath} focusReportPath={focusReportPath} focusComparePaths={focusComparePaths} onFocusHandled={() => { setFocusRunPath(null); setFocusReportPath(null); setFocusComparePaths([]); }} onRouteChange={reflectLibrarySelection} /> : view === "budget" ? <BudgetView data={data} /> : view === "data" ? <DataSources data={data} onRefresh={refresh} /> : <Backends data={data} />}
        </section>
      </div>
      <nav className="studio-mobile-dock lg:hidden" aria-label="Research workspace">{navItems.map((item) => { const Icon = item.icon; const active = view === item.id; const label = item.id === "new" ? "New experiment" : item.id === "runs" ? "Runs" : item.id === "data" ? "Data" : item.label; return <button key={item.id} type="button" aria-label={label} aria-current={active ? "page" : undefined} className={`studio-mobile-dock-item ${active ? "studio-mobile-dock-item-active" : ""}`} onClick={() => navigate(item.id)}><span className="studio-mobile-dock-icon"><Icon /></span><span className="studio-mobile-dock-label">{label}</span></button>; })}</nav>
      <Dialog open={docsOpen} onOpenChange={setDocsOpen}><DialogContent className="sm:max-w-lg"><DialogHeader><DialogTitle>Using P2P Studio</DialogTitle><DialogDescription>This interface is a local companion to the CLI.</DialogDescription></DialogHeader><div className="space-y-4 text-sm leading-6"><div className="flex gap-3"><FolderOpen className="mt-1 size-4 shrink-0 text-primary" /><p><strong>Create or revisit work.</strong> Experiments run through the same Python command handlers as the CLI, while the library keeps runs in the current and the earlier format together.</p></div><div className="flex gap-3"><BarChart3 className="mt-1 size-4 shrink-0 text-primary" /><p><strong>Inspect artifacts.</strong> Runs, reports, and budgets read compatible records under <code className="rounded bg-muted px-1">outputs/</code> without rewriting their sources.</p></div><div className="flex gap-3"><ShieldCheck className="mt-1 size-4 shrink-0 text-primary" /><p><strong>Keep it local.</strong> The server binds to 127.0.0.1 by default and includes no publishing path.</p></div></div><DialogFooter><Button variant="outline" onClick={() => { setDocsOpen(false); setWelcomeOpen(true); }}><Play /> Replay welcome</Button><Button onClick={() => setDocsOpen(false)}>Done</Button></DialogFooter></DialogContent></Dialog>
      {data && <WelcomeTour open={welcomeOpen} data={data} onFinish={finishWelcome} onExperiment={() => navigate("new")} onDemo={() => void runOfflineDemo()} />}
    </main>
  );
}
