"use client";

import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  ArrowLeft,
  BarChart3,
  ChevronRight,
  Download,
  FileImage,
  FileText,
  Gauge,
  GitCompareArrows,
  History,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  Save,
  ScanLine,
  Search,
  Sparkles,
  Star,
  Tag,
  TriangleAlert,
  Users,
  Weight,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { populationFamily as modeFamily, populationFamilyColors as modeFamilies } from "@/app/population-style";

type Metric = { metric: string; value: number; model?: string; split?: string; unit?: string; predictionPolicy?: string; qidsRequested?: number; qidsUsed?: number };
type Annotation = { label?: string; pinned?: boolean; updatedAt?: string };
type PopulationSummary = {
  generated: number;
  selected: number;
  antiTypes: number;
  sourcePath: string;
  modes: { mode: string; family?: string; generated: number; selected: number; antiTypes: number }[];
};
type Question = {
  qid: string;
  question: string;
  split: string;
  option_order: string[];
  target: Record<string, number>;
  prediction: Record<string, number>;
};
type Run = {
  path: string;
  runId: string;
  seed?: number;
  method?: string;
  conditions?: Record<string, unknown>;
  metrics: Metric[];
  usage: Record<string, number>;
  questionCount: number;
  hasDiagnostics: boolean;
  population?: PopulationSummary | null;
  modifiedAt: string;
  origin: "current" | "legacy";
  capabilities?: { metrics?: boolean; predictions?: boolean; weights?: boolean; generation?: boolean; folds?: boolean; usage?: boolean; reports?: boolean; export?: boolean; reevaluate?: boolean };
  provenance?: { label?: string; sourceRepository?: string; sourcePath?: string; cleaningContract?: string };
  sourceFormat?: string;
  sourceVersion?: number;
  compatibilityWarnings?: string[];
  annotation?: Annotation;
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
  annotation?: Annotation;
};
type HealthCheck = { level: "ok" | "info" | "warning"; label: string; detail: string };
type RunHealth = { status: "healthy" | "attention"; warningCount: number; checks: HealthCheck[]; facts: { label: string; value: string; hint?: string }[] };
type MetricInfo = { id: string; label: string; short: string; direction: "lower" | "higher"; unit: string; description: string };
type AggregateSummary = {
  path: string; name: string; status: string; runCount?: number; includedRunCount?: number; metricRows?: number;
  groupKeys: string[]; groupCount: number; metrics: string[]; center?: string; spread?: string; digest?: string; modifiedAt: string;
};
type AggregateRow = Record<string, string | number | null> & { metric: string; center: number | null; spread: number | null; n: number; missing_count?: number };
type AggregateDetail = { summary: AggregateSummary; rows: AggregateRow[]; checks: Record<string, string>; missingRows: unknown[]; missingPolicy?: string; files: string[] };
type RunDetail = {
  model?: { method?: string; solver?: string; status?: string; coefficients?: Record<string, number> };
  health?: RunHealth;
  questions?: Question[];
  metrics?: { metrics?: Metric[] };
  usage?: { summary?: Record<string, number>; records?: Record<string, unknown>[] };
  legacy?: { notes?: string[]; archivedImages?: string[] };
  endowmentRows?: Record<string, string>[];
  modeRows?: Record<string, string>[];
  modeSelectionSource?: "summary_by_mode" | "derived_coefficients" | "unavailable";
  modeSummaryRows?: Record<string, string>[];
  modeEntropyRows?: Record<string, string>[];
  aggregateEntropy?: Record<string, string>[];
  archive?: ReportDetail;
  diagnosticsAvailable?: boolean;
  notes?: string[];
  summary?: Run;
  annotation?: Annotation;
  job?: { id: string; command: string; config: string; output: string; status: string; createdAt: string };
};
type ReportDetail = {
  summary: Report;
  documents: { name: string; title: string; content: string }[];
  images: { name: string; path: string; category: "predictions" | "diagnostics" | "generation" | "weights" | "other"; size: number }[];
  metadata: Record<string, unknown>;
  results: Record<string, unknown>;
  tokenUsage: { total_cost?: number; per_model_module?: Record<string, Record<string, number>> };
  metricRows: Record<string, string>[];
  endowmentRows: Record<string, string>[];
  modeRows: Record<string, string>[];
  modeSelectionSource?: "summary_by_mode" | "derived_coefficients" | "unavailable";
  modeSummaryRows: Record<string, string>[];
  modeEntropyRows: Record<string, string>[];
  entropyRows: Record<string, string>[];
  aggregateEntropy: Record<string, string>[];
  files: { name: string; size: number; type: string }[];
};
type ResultsData = {
  runs: Run[];
  reports: Report[];
  workspace: { counts: { runs: number; currentRuns: number; historicalRuns: number; reports: number; aggregates?: number } };
  annotations?: Record<string, Annotation>;
  capabilities?: { metrics?: MetricInfo[] };
};

// Used only when a workspace's capabilities are unavailable (older server).
const FALLBACK_METRICS: MetricInfo[] = [
  { id: "mse", label: "Raw option-level MSE", short: "MSE", direction: "lower", unit: "binary option", description: "" },
  { id: "distribution_mse", label: "Rescaled MSE", short: "Rescaled MSE", direction: "lower", unit: "binary option", description: "" },
  { id: "one_minus_jsd", label: "1 − JSD", short: "1 − JSD", direction: "higher", unit: "question", description: "" },
  { id: "emd", label: "EMD / WD", short: "EMD", direction: "lower", unit: "question", description: "" },
  { id: "mcpa", label: "MCPA", short: "MCPA", direction: "higher", unit: "question", description: "" },
];

function metricCatalog(data?: { capabilities?: { metrics?: MetricInfo[] } }): MetricInfo[] {
  return data?.capabilities?.metrics?.length ? data.capabilities.metrics : FALLBACK_METRICS;
}

function metricInfo(catalog: MetricInfo[], id: string): MetricInfo {
  return catalog.find((item) => item.id === id) || { id, label: id.replaceAll("_", " "), short: id.replaceAll("_", " "), direction: "lower", unit: "", description: "" };
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body as T;
}

async function postJson<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload as T;
}

function formatNumber(value?: number, digits = 0) {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function formatMoney(value?: number) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(new Date(value));
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function formatModelName(value?: string) {
  if (!value) return "Model unavailable";
  const normalized = value.trim().toLowerCase().replaceAll("_", "").replaceAll("-", "");
  if (normalized === "elasticnet") return "Elastic Net";
  if (normalized === "lasso") return "Lasso";
  return value.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricValue(run: Run, metric: string) {
  return run.metrics.find((item) => item.metric === metric && item.split === "test")?.value
    ?? [...run.metrics].reverse().find((item) => item.metric === metric)?.value;
}

function downloadText(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function RunComparison({ runs, catalog, onRemove, onClear }: { runs: Run[]; catalog: MetricInfo[]; onRemove: (path: string) => void; onClear: () => void }) {
  const conditionKeys = [...new Set(runs.flatMap((run) => Object.keys(run.conditions || {})))].filter((key) => key !== "method").sort();
  const recorded = new Set(runs.flatMap((run) => run.metrics.map((metric) => metric.metric)));
  const rows: (string | number)[][] = [
    ["Method", ...runs.map((run) => formatModelName(run.method))],
    ["Seed", ...runs.map((run) => run.seed ?? "—")],
    ...catalog.filter((metric) => recorded.has(metric.id)).map((metric) => [`${metric.short} ${metric.direction === "lower" ? "↓" : "↑"} (test)`, ...runs.map((run) => formatNumber(metricValue(run, metric.id), 4))]),
    ["Input tokens", ...runs.map((run) => formatNumber(run.usage.input_tokens))],
    ["Output tokens", ...runs.map((run) => formatNumber(run.usage.output_tokens))],
    ["Cost", ...runs.map((run) => formatMoney(run.usage.cost_usd))],
    ["Lineage", ...runs.map((run) => run.origin)],
    ...conditionKeys.map((key) => [`Config · ${key}`, ...runs.map((run) => {
      const value = run.conditions?.[key];
      return typeof value === "object" && value !== null ? JSON.stringify(value) : value == null ? "—" : String(value);
    })]),
  ];
  const exportCsv = () => {
    const quote = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const content = [["Field", ...runs.map((run) => run.runId)], ...rows].map((row) => row.map(quote).join(",")).join("\n");
    downloadText("p2p-run-comparison.csv", `${content}\n`, "text/csv;charset=utf-8");
  };
  return <section className="surface-card mb-5 overflow-hidden"><div className="flex flex-col gap-3 border-b px-5 py-4 sm:flex-row sm:items-center sm:justify-between"><div><div className="flex items-center gap-2"><GitCompareArrows className="size-4 text-primary" /><h2 className="section-title">Run comparison</h2></div><p className="section-copy">Test metrics and execution footprint for {runs.length} selected records.</p></div><div className="flex gap-2"><Button variant="outline" size="sm" onClick={exportCsv}><Download /> Export CSV</Button><Button variant="ghost" size="sm" onClick={onClear}>Clear</Button></div></div><div className="overflow-x-auto"><table className="w-full min-w-[760px] text-sm"><thead className="bg-muted/35 text-left"><tr><th className="px-5 py-3 text-xs uppercase tracking-[.06em] text-muted-foreground">Field</th>{runs.map((run) => <th key={run.path} className="min-w-48 px-5 py-3"><span className="flex items-center justify-between gap-2"><span className="min-w-0"><strong className="block truncate">{run.runId}</strong><span className="block truncate font-mono text-[10px] font-normal text-muted-foreground">{run.path}</span></span><button aria-label={`Remove ${run.runId} from comparison`} className="rounded p-1 text-muted-foreground hover:bg-muted" onClick={() => onRemove(run.path)}><X className="size-3.5" /></button></span></th>)}</tr></thead><tbody>{rows.map((row) => <tr key={String(row[0])} className="border-t"><th className="px-5 py-3 text-left text-xs font-medium text-muted-foreground">{row[0]}</th>{row.slice(1).map((value, index) => <td key={`${String(row[0])}-${runs[index].path}`} className="px-5 py-3 font-mono text-xs">{String(value)}</td>)}</tr>)}</tbody></table></div></section>;
}

function artifactUrl(path: string) {
  return `/api/artifact?path=${encodeURIComponent(path)}`;
}

function fileUrl(path: string) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

function AnnotationEditor({ path, annotation, onSaved }: { path: string; annotation?: Annotation; onSaved: (annotation: Annotation) => void }) {
  const [label, setLabel] = useState(annotation?.label || "");
  const [saving, setSaving] = useState(false);
  const [editing, setEditing] = useState(false);
  useEffect(() => { setLabel(annotation?.label || ""); }, [path, annotation?.label]);
  async function save(pinned = Boolean(annotation?.pinned)) {
    setSaving(true);
    try {
      const result = await postJson<{ annotation: Annotation }>("/api/annotations", { path, label, pinned });
      onSaved(result.annotation);
      toast.success(pinned ? "Record pinned to the research library." : label ? "Research label saved." : "Annotation removed.");
      setEditing(false);
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not save this annotation."); }
    finally { setSaving(false); }
  }
  return <section className="record-annotation mb-5"><span className="record-annotation-icon"><Tag /></span>{editing ? <div className="record-annotation-editor"><label htmlFor="record-label">Research label</label><Input id="record-label" autoFocus value={label} maxLength={80} onChange={(event) => setLabel(event.target.value)} placeholder="e.g. final candidate" /><div className="flex gap-2"><Button variant="ghost" size="sm" onClick={() => { setLabel(annotation?.label || ""); setEditing(false); }}>Cancel</Button><Button size="sm" onClick={() => void save()} disabled={saving}>{saving ? <LoaderCircle className="animate-spin" /> : <Save />} Save label</Button></div></div> : <button type="button" className="record-annotation-label" onClick={() => setEditing(true)}><small>Research label</small><strong>{annotation?.label || "Add a memorable label"}</strong></button>}<Button variant="ghost" size="sm" onClick={() => void save(!annotation?.pinned)} disabled={saving}><Star className={annotation?.pinned ? "fill-current" : ""} />{annotation?.pinned ? "Pinned" : "Pin"}</Button></section>;
}

function Header({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return <div className="page-heading mb-6"><p className="page-eyebrow mb-2">{eyebrow}</p><h1 className="text-2xl font-semibold tracking-[-0.035em] sm:text-[2rem]">{title}</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">{copy}</p></div>;
}

function Loading({ label }: { label: string }) {
  return <div className="loading-record" aria-live="polite"><div className="loading-record-line loading-record-kicker" /><div className="loading-record-line loading-record-title" /><div className="loading-record-line loading-record-copy" /><div className="loading-record-grid">{[0,1,2,3].map((item) => <span key={item} />)}</div><p>{label}</p></div>;
}

function Tabs<T extends string>({ value, items, onChange, sticky = false }: { value: T; items: { id: T; label: string; count?: number }[]; onChange: (value: T) => void; sticky?: boolean }) {
  return <div className={`result-tabs mb-5 flex flex-wrap gap-1 rounded-xl border bg-card p-1 ${sticky ? "result-tabs-sticky" : ""}`}>{items.map((item) => <button key={item.id} type="button" aria-pressed={value === item.id} onClick={() => onChange(item.id)} className={`shrink-0 whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${value === item.id ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}>{item.label}{item.count != null && <span className={`ml-2 font-mono text-xs ${value === item.id ? "text-white/70" : "text-muted-foreground"}`}>{item.count}</span>}</button>)}</div>;
}

function humanizeRunId(value: string) {
  const cleaned = value
    .replace(/[-_]r\d{3,}$/i, "")
    .replace(/stage[_-]?2/gi, "Stage 2")
    .replace(/[_-]+/g, " ")
    .replace(/\bW(\d+)\b/g, "Wave $1")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^Stage 2 ablation\s+/i, "Stage 2 ablation · ")
    .replace(/\s+(Wave \d+)$/i, " · $1");
  if (!cleaned) return value;
  return cleaned.split(" · ").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" · ");
}

function runFamily(run: Run) {
  const key = run.runId
    .replace(/(?:[_-]W\d+)?[-_]r\d{3,}$/i, "")
    .replace(/[_-]W\d+$/i, "")
    .replace(/[-_]repeat[-_]?\d+$/i, "") || run.runId;
  const wave = run.runId.match(/(?:^|[_-])W(\d+)/i)?.[1];
  const repeat = run.runId.match(/[-_]r0*(\d+)$/i)?.[1] || (run.conditions?.repeat != null ? String(run.conditions.repeat) : undefined);
  return { key, label: humanizeRunId(key), instance: [wave ? `W${wave}` : "", repeat ? `r${repeat}` : ""].filter(Boolean).join(" · ") };
}

function RunPopulationInline({ population }: { population?: PopulationSummary | null }) {
  if (!population?.generated) return null;
  const families = [...new Set(population.modes.map((item) => modeFamily(item.mode, item.family)))];
  return <span className="run-population-inline" title={`Population source: ${population.sourcePath}`}><span className="run-population-key" aria-label={`Template families present: ${families.join(", ")}`}>{families.map((family) => <i key={family} title={family} aria-label={family} style={{ backgroundColor: modeFamilies[family] }} />)}</span><span className="font-mono">{formatNumber(population.generated)} → {formatNumber(population.selected)}</span><small>generated → selected</small>{population.antiTypes > 0 && <span className="run-population-anti">{population.antiTypes} anti</span>}</span>;
}

function PopulationFamilyLegend() {
  return <div className="run-population-legend" aria-label="Population colour legend"><strong>Population colours</strong>{Object.entries(modeFamilies).map(([family, color]) => <span key={family}><i style={{ backgroundColor: color }} />{family}</span>)}<span className="run-anti-key"><i />Anti-type</span><small>Dots identify template families; stripes consistently mark negative coefficients.</small></div>;
}

function RunLibraryRow({ run, annotation, selected, onCompare, onOpen }: { run: Run; annotation?: Annotation; selected: boolean; onCompare: () => void; onOpen: () => void }) {
  const family = runFamily(run);
  const compact = [
    ["Method", run.method || "—"],
    ["MSE", formatNumber(metricValue(run, "mse"), 3)],
    ["1 − JSD", formatNumber(metricValue(run, "one_minus_jsd"), 3)],
    ["EMD", formatNumber(metricValue(run, "emd"), 3)],
    ["MCPA", formatNumber(metricValue(run, "mcpa"), 3)],
    ["Updated", formatDate(run.modifiedAt)],
  ];
  return (
    <div className={`run-family-row flex items-stretch ${selected ? "bg-primary/5" : "hover:bg-muted/45"}`}>
      <label className="flex w-12 shrink-0 items-start justify-center pt-5" title="Select for comparison"><input type="checkbox" aria-label={`Compare ${run.runId}`} checked={selected} onChange={onCompare} className="size-4 accent-[var(--primary)]" /></label>
      <button onClick={onOpen} className="grid min-w-0 flex-1 gap-3 px-2 py-4 text-left transition-colors lg:grid-cols-[minmax(260px,1.6fr)_120px_repeat(4,90px)_100px_24px] lg:items-center lg:gap-3 lg:pr-5">
        <span className="min-w-0"><span className="flex min-w-0 items-center gap-2"><strong className="truncate text-sm">{run.runId}</strong>{annotation?.pinned && <Star className="size-3.5 shrink-0 fill-current text-amber-500" />}{run.origin === "legacy" && <Badge variant="outline" className="shrink-0 text-[10px]">earlier format</Badge>}{family.instance && <Badge variant="outline" className="shrink-0 font-mono text-[10px]">{family.instance}</Badge>}{annotation?.label && <Badge variant="secondary" className="max-w-40 truncate text-[10px]">{annotation.label}</Badge>}</span><span className="block truncate font-mono text-[11px] text-muted-foreground">{run.path}</span><RunPopulationInline population={run.population} /></span>
        <span className="hidden text-sm lg:block">{run.method || "—"}</span>
        {["mse", "one_minus_jsd", "emd", "mcpa"].map((name) => <span key={name} className="hidden font-mono text-sm lg:block">{formatNumber(metricValue(run, name), 3)}</span>)}
        <span className="hidden text-xs text-muted-foreground lg:block">{formatDate(run.modifiedAt)}</span><ChevronRight className="hidden size-4 text-muted-foreground lg:block" />
        <span className="grid grid-cols-2 gap-x-5 gap-y-2 sm:grid-cols-3 lg:hidden">{compact.map(([label, value]) => <span key={label}><span className="block text-[10px] font-semibold uppercase tracking-[.06em] text-muted-foreground">{label}</span><span className="mt-0.5 block font-mono text-xs">{value}</span></span>)}</span>
      </button>
    </div>
  );
}

function MetricBars({ metrics, catalog }: { metrics: Metric[]; catalog: MetricInfo[] }) {
  return <div className="grid gap-3 sm:grid-cols-2">{metrics.map((metric) => { const info = metricInfo(catalog, metric.metric); const meta = { label: info.label, direction: info.direction === "lower" ? "Lower is better" : "Higher is better" }; return <article key={`${metric.metric}-${metric.split}`} className="rounded-xl border bg-background p-4" title={info.description}><div className="flex items-start justify-between gap-3"><div><p className="text-sm font-semibold">{meta.label}</p><p className="mt-1 text-[11px] text-muted-foreground">{meta.direction} · {metric.split || "unspecified split"}</p></div><p className="font-mono text-xl font-semibold">{formatNumber(metric.value, 4)}</p></div><div className="mt-4 flex flex-wrap gap-1.5"><Badge variant="outline" className="font-normal">{metric.unit || "unit not recorded"}</Badge><Badge variant="outline" className="font-normal">{metric.predictionPolicy || "policy not recorded"}</Badge>{metric.qidsRequested != null && <Badge variant="secondary" className="font-normal">{metric.qidsUsed ?? 0}/{metric.qidsRequested} QIDs</Badge>}</div></article>; })}</div>;
}

function Distribution({ question }: { question: Question }) {
  return <div className="space-y-3"><div className="flex items-start justify-between gap-4"><div><p className="font-semibold">{question.qid}</p><p className="mt-1 text-sm leading-5 text-muted-foreground">{question.question}</p></div><Badge variant="outline">{question.split}</Badge></div>{question.option_order.map((option) => <div key={option} className="grid grid-cols-[minmax(90px,180px)_1fr_52px] items-center gap-3 text-xs"><span className="truncate" title={option}>{option}</span><div className="space-y-1"><div className="viz-target-track h-2 rounded-full"><div className="viz-target-fill h-full rounded-full" style={{ width: `${Math.max(0, question.target[option] || 0) * 100}%` }} /></div><div className="viz-prediction-track h-2 rounded-full"><div className="viz-prediction-fill h-full rounded-full" style={{ width: `${Math.max(0, question.prediction[option] || 0) * 100}%` }} /></div></div><span className="text-right font-mono">{((question.prediction[option] || 0) * 100).toFixed(0)}%</span></div>)}<div className="flex gap-4 text-xs text-muted-foreground"><span><i className="viz-target-fill mr-1 inline-block size-2 rounded-full" /> Target</span><span><i className="viz-prediction-fill mr-1 inline-block size-2 rounded-full" /> Prediction</span></div></div>;
}

const predictionSplits = [
  { id: "train", label: "Train", color: "var(--chart-1)", role: "Fits the proxy ensemble", tone: "train" },
  { id: "valid", label: "Valid", color: "var(--chart-2)", role: "Tunes and selects the model", tone: "valid" },
  { id: "test", label: "Test", color: "var(--chart-3)", role: "Checks held-out validity", tone: "test" },
] as const;

function SplitRoleLegend() {
  return <div className="mb-5 grid gap-2 sm:grid-cols-3" aria-label="Dataset split roles">{predictionSplits.map((split) => <div key={split.id} className={`split-role-card split-role-${split.tone}`}><div className="flex items-center gap-2"><i className={`split-swatch-${split.tone} size-2.5 rounded-full`} /><strong className="text-sm">{split.label}</strong></div><p className="mt-1 text-xs text-muted-foreground">{split.role}</p></div>)}</div>;
}

function PredictionFacet({ split, label, color, questions, selectedQid, onSelect }: { split: string; label: string; color: string; questions: Question[]; selectedQid?: string; onSelect: (qid: string) => void }) {
  const splitQuestions = questions.filter((question) => question.split === split);
  const points = splitQuestions.flatMap((question) => question.option_order.map((option) => ({ qid: question.qid, option, x: question.target[option] || 0, y: question.prediction[option] || 0 })));
  const ticks = [0, .5, 1];
  return <figure className="min-w-0"><figcaption className="mb-2 flex items-center justify-between gap-3 text-sm"><span className="font-semibold">{label}</span><span className="font-mono text-xs text-muted-foreground">{splitQuestions.length} questions</span></figcaption><svg viewBox="0 0 352 290" className="block h-auto w-full" role="img" aria-label={`${label} predicted share versus target share`}><rect x="48" y="18" width="282" height="220" fill="var(--background)" stroke="var(--border)" /><line x1="48" y1="238" x2="330" y2="18" stroke="var(--muted-foreground)" strokeDasharray="5 5" opacity=".55" />{ticks.map((tick) => <g key={tick}><line x1={48 + tick * 282} y1="238" x2={48 + tick * 282} y2="244" stroke="var(--muted-foreground)" /><text x={48 + tick * 282} y="260" textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"} fill="var(--muted-foreground)" fontSize="12">{tick.toFixed(1)}</text><line x1="42" y1={238 - tick * 220} x2="48" y2={238 - tick * 220} stroke="var(--muted-foreground)" /><text x="38" y={242 - tick * 220} textAnchor="end" fill="var(--muted-foreground)" fontSize="12">{tick.toFixed(1)}</text></g>)}{points.map((point, index) => <circle key={`${point.qid}-${point.option}-${index}`} onClick={() => onSelect(point.qid)} cx={48 + point.x * 282} cy={238 - point.y * 220} r={point.qid === selectedQid ? 5 : 3.4} fill={color} opacity={point.qid === selectedQid ? 1 : .68} stroke={point.qid === selectedQid ? "var(--foreground)" : "none"} strokeWidth="1" className="cursor-pointer"><title>{point.qid} · {point.option}: target {point.x.toFixed(3)}, prediction {point.y.toFixed(3)}</title></circle>)}<text x="189" y="286" textAnchor="middle" fill="var(--viz-target)" fontSize="12" fontWeight="600">Target share</text><text x="13" y="128" transform="rotate(-90 13 128)" textAnchor="middle" fill="var(--viz-prediction)" fontSize="12" fontWeight="600">Predicted share</text>{!points.length && <text x="189" y="132" textAnchor="middle" fill="var(--muted-foreground)" fontSize="12">No {label.toLowerCase()} questions</text>}</svg></figure>;
}

function PredictionExplorer({ questions }: { questions: Question[] }) {
  const [qid, setQid] = useState(questions[0]?.qid || "");
  const selected = questions.find((question) => question.qid === qid) || questions[0];
  if (!questions.length) return <p className="text-sm text-muted-foreground">No question-level diagnostics were written for this run.</p>;
  return <div className="space-y-5"><section className="surface-card p-5 sm:p-6"><div className="mb-5"><h2 className="section-title">Prediction calibration by split</h2><p className="section-copy">Train, validation, and test are shown separately on the same scale. Select a mark to inspect its full question distribution.</p></div><SplitRoleLegend /><div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">{predictionSplits.map((item) => <PredictionFacet key={item.id} split={item.id} label={item.label} color={item.color} questions={questions} selectedQid={selected?.qid} onSelect={setQid} />)}</div></section><section className="surface-card p-5 sm:p-6"><div className="mx-auto max-w-3xl"><div className="mb-4"><label className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground" htmlFor="question-picker">Question drill-down</label><select id="question-picker" className="mt-2 h-10 w-full rounded-md border bg-background px-3 text-sm" value={selected?.qid} onChange={(event) => setQid(event.target.value)}>{predictionSplits.map((split) => <optgroup key={split.id} label={split.label}>{questions.filter((question) => question.split === split.id).map((question) => <option key={question.qid} value={question.qid}>{question.qid} · {question.question}</option>)}</optgroup>)}</select></div>{selected && <Distribution question={selected} />}</div></section></div>;
}

function normalizedEntropy(values: number[]) {
  const positive = values.filter((value) => value > 0);
  if (values.length < 2 || !positive.length) return 0;
  return -positive.reduce((sum, value) => sum + value * Math.log2(value), 0) / Math.log2(values.length);
}

function EntropyExplorer({ questions }: { questions: Question[] }) {
  const rows = useMemo(() => questions.map((question) => ({ question, target: normalizedEntropy(question.option_order.map((option) => question.target[option] || 0)), prediction: normalizedEntropy(question.option_order.map((option) => question.prediction[option] || 0)) })).sort((a, b) => b.prediction - a.prediction), [questions]);
  const [qid, setQid] = useState(rows[0]?.question.qid || "");
  const selected = rows.find((row) => row.question.qid === qid) || rows[0];
  return <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]"><section className="surface-card p-5 sm:p-6"><div className="mb-4"><h2 className="section-title">Question entropy explorer</h2><p className="section-copy">Normalized response uncertainty for each question.</p></div><div className="max-h-[560px] space-y-2 overflow-auto pr-2">{rows.slice(0, 60).map((row) => <button key={row.question.qid} className={`grid w-full grid-cols-[110px_1fr_48px] items-center gap-3 rounded-lg px-2 py-2 text-left text-xs ${selected?.question.qid === row.question.qid ? "bg-primary/8" : "hover:bg-muted"}`} onClick={() => setQid(row.question.qid)}><span className="truncate font-mono">{row.question.qid}</span><span className="relative h-3 rounded-full bg-muted"><i className="viz-target-fill absolute inset-y-0 left-0 rounded-full" style={{ width: `${row.target * 100}%` }} /><i className="viz-prediction-fill absolute inset-y-[3px] left-0 rounded-full" style={{ width: `${row.prediction * 100}%` }} /></span><span className="text-right font-mono">{row.prediction.toFixed(2)}</span></button>)}</div><div className="mt-4 flex gap-4 text-xs text-muted-foreground"><span><i className="viz-target-fill mr-1 inline-block size-2 rounded-full" /> Target</span><span><i className="viz-prediction-fill mr-1 inline-block size-2 rounded-full" /> Prediction</span></div></section><section className="surface-card p-5 sm:p-6">{selected && <Distribution question={selected.question} />}</section></div>;
}

function Coefficients({ coefficients = {}, endowments = [] }: { coefficients?: Record<string, number>; endowments?: Record<string, string>[] }) {
  const rows = Object.entries(coefficients).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 30);
  const maximum = Math.max(...rows.map(([, value]) => Math.abs(value)), 1e-9);
  const endowmentById = useMemo(() => new Map(endowments.filter((row) => row.eid).map((row) => [row.eid, row])), [endowments]);
  const [selectedName, setSelectedName] = useState(rows[0]?.[0] || "");
  const selectedRow = rows.find(([name]) => name === selectedName) || rows[0];
  const selectedEndowment = selectedRow ? endowmentById.get(selectedRow[0]) : undefined;
  const attributes = (selectedEndowment?.attributes || "").split(/;\s*/).map((item) => item.trim()).filter(Boolean);
  if (!rows.length) return <p className="text-sm text-muted-foreground">No fitted coefficients were written.</p>;
  return <section className="surface-card p-5 sm:p-6"><div className="mb-5"><h2 className="section-title">Proxy weights</h2><p className="section-copy">Thirty largest coefficients by absolute magnitude. Select a proxy to inspect the endowment used to generate its responses. Stripes mark negative anti-types.</p></div><div className="grid gap-5 xl:grid-cols-[1.25fr_.85fr]"><div className="min-w-0 overflow-hidden rounded-xl border bg-background"><div className="grid grid-cols-[minmax(120px,1fr)_2fr_72px] gap-3 border-b bg-muted/35 px-3 py-2 text-[10px] font-semibold uppercase tracking-[.06em] text-muted-foreground"><span>Proxy</span><span>Relative weight</span><span className="text-right">Coefficient</span></div><div className="max-h-[620px] overflow-auto">{rows.map(([name, value]) => <button key={name} type="button" aria-pressed={selectedRow?.[0] === name} onClick={() => setSelectedName(name)} className={`grid w-full grid-cols-[minmax(120px,1fr)_2fr_72px] items-center gap-3 border-b px-3 py-3 text-left text-xs last:border-0 ${selectedRow?.[0] === name ? "bg-primary/8" : "hover:bg-muted/50"}`}><span className="truncate font-mono" title={name}>{name}</span><span className="h-2 overflow-hidden rounded-full bg-muted"><i className={`block h-full rounded-full ${value < 0 ? "coefficient-anti-fill" : "viz-prediction-fill"}`} style={{ width: `${Math.abs(value) / maximum * 100}%` }} /></span><span className="text-right font-mono">{value.toFixed(4)}</span></button>)}</div></div><aside className="self-start rounded-xl border bg-background p-5 xl:sticky xl:top-4"><p className="page-eyebrow mb-2">Selected proxy</p><h3 className="break-words font-mono text-sm font-semibold">{selectedRow?.[0]}</h3><div className="mt-3 flex flex-wrap gap-1.5">{selectedEndowment?.role && <Badge variant="secondary">{selectedEndowment.role}</Badge>}{selectedEndowment?.mode && <Badge variant="outline">{selectedEndowment.mode}</Badge>}</div>{selectedEndowment ? <><div className="mt-5"><p className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">Endowment</p><p className="mt-2 text-sm leading-6">{selectedEndowment.endow_text || "No narrative endowment text was stored for this proxy."}</p></div>{attributes.length > 0 && <div className="mt-5"><p className="mb-2 text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">Attributes</p><div className="flex max-h-44 flex-wrap gap-1.5 overflow-auto">{attributes.map((attribute) => <span key={attribute} className="rounded-md bg-muted px-2 py-1 text-[11px]">{attribute}</span>)}</div></div>}</> : <p className="mt-5 text-sm leading-6 text-muted-foreground">No endowment metadata was stored for this proxy. Its fitted coefficient is still available in the run diagnostics.</p>}</aside></div></section>;
}

type EnsembleProxy = {
  id: string;
  coefficient: number;
  magnitude: number;
  selected: boolean;
  antiType: boolean;
  mode: string;
  templateFamily: ModeFamily;
  generationStep: string;
  role: string;
  text: string;
  attributes: string[];
  scale: number;
};

function questionMse(question: Question) {
  if (!question.option_order.length) return 0;
  return question.option_order.reduce((total, option) => {
    const residual = (question.prediction[option] || 0) - (question.target[option] || 0);
    return total + residual * residual;
  }, 0) / question.option_order.length;
}

function EnsembleEvidenceView({
  questions,
  coefficients = {},
  endowments = [],
  modelMethod,
  availableModels = [],
}: {
  questions: Question[];
  coefficients?: Record<string, number>;
  endowments?: Record<string, string>[];
  modelMethod?: string;
  availableModels?: string[];
}) {
  const evidenceQuestions = useMemo(() => {
    const heldOut = questions.filter((question) => question.split === "test");
    return heldOut.length ? heldOut : questions;
  }, [questions]);
  const endowmentById = useMemo(
    () => new Map(endowments.filter((row) => row.eid).map((row) => [row.eid, row])),
    [endowments],
  );
  const proxies = useMemo<EnsembleProxy[]>(() => {
    const endowmentIds = endowments.map((row) => row.eid).filter(Boolean);
    const coefficientIds = Object.keys(coefficients);
    // A reference respondent is a fitted model anchor, not a generated proxy.
    // Keep coefficient-only terms out of a real Stage-1 population formation.
    // Coefficient-only records still fall back to their coefficients.
    const ids = endowmentIds.length ? endowmentIds : coefficientIds;
    const uniqueIds = [...new Set(ids)];
    const weighted = uniqueIds.map((id) => {
      const endowment = endowmentById.get(id);
      const storedWeight = Number(endowment?.weight);
      const fittedWeight = coefficients[id];
      const coefficient = Number.isFinite(fittedWeight) ? fittedWeight : Number.isFinite(storedWeight) ? storedWeight : 0;
      return { id, coefficient };
    });
    const maximum = Math.max(...weighted.map(({ coefficient }) => Math.abs(coefficient)), 1e-9);
    return weighted.map(({ id, coefficient }) => {
      const endowment = endowmentById.get(id);
      const mode = endowment?.mode || "unclassified";
      const magnitude = Math.abs(coefficient);
      const selected = magnitude > 1e-10;
      const generationStep = endowment?.update_step || endowment?.generation_step || endowment?.step || endowment?.round || "not recorded";
      return {
        id,
        coefficient,
        magnitude,
        selected,
        antiType: coefficient < 0,
        mode,
        templateFamily: modeFamily(mode),
        generationStep,
        role: endowment?.role || "proxy",
        text: endowment?.endow_text || "No narrative endowment text was stored for this proxy.",
        attributes: (endowment?.attributes || "").split(/;\s*/).map((item) => item.trim()).filter(Boolean),
        scale: selected ? .88 + magnitude / maximum * .62 : .72,
      };
    });
  }, [coefficients, endowments, endowmentById]);
  const auxiliaryTerms = useMemo(() => Object.entries(coefficients)
    .filter(([id, coefficient]) => !endowmentById.has(id) && Math.abs(coefficient) > 1e-10)
    .map(([id, coefficient]) => ({ id, coefficient })), [coefficients, endowmentById]);
  const referenceRespondent = auxiliaryTerms.find((term) => term.id === "__reference_respondent__");
  const templateCounts = useMemo(() => Object.keys(modeFamilies).map((family) => ({ family: family as ModeFamily, count: proxies.filter((proxy) => proxy.templateFamily === family).length })).filter((item) => item.count > 0), [proxies]);
  const formationColumns = Math.max(1, Math.ceil(Math.sqrt(proxies.length)));
  const selectedProxies = useMemo(() => proxies.filter((proxy) => proxy.selected).sort((left, right) => right.magnitude - left.magnitude), [proxies]);
  const antiTypes = useMemo(() => selectedProxies.filter((proxy) => proxy.antiType), [selectedProxies]);
  const selectedTermCount = selectedProxies.length + auxiliaryTerms.length;
  const [view, setView] = useState<"ensemble" | "evidence">(proxies.length ? "ensemble" : "evidence");
  const [selectedProxyId, setSelectedProxyId] = useState(selectedProxies[0]?.id || proxies[0]?.id || "");
  const [selectedQid, setSelectedQid] = useState(evidenceQuestions[0]?.qid || "");
  const selectedProxy = proxies.find((proxy) => proxy.id === selectedProxyId) || proxies[0];
  useEffect(() => {
    if (!proxies.some((proxy) => proxy.id === selectedProxyId)) setSelectedProxyId(selectedProxies[0]?.id || proxies[0]?.id || "");
  }, [proxies, selectedProxies, selectedProxyId]);
  const selectedQuestion = evidenceQuestions.find((question) => question.qid === selectedQid) || evidenceQuestions[0];
  const visibleQuestions = evidenceQuestions.slice(0, 14);
  const selectedResiduals = selectedQuestion?.option_order.map((option) => ({
    option,
    value: (selectedQuestion.prediction[option] || 0) - (selectedQuestion.target[option] || 0),
  })) || [];
  const largestResidual = [...selectedResiduals].sort((left, right) => Math.abs(right.value) - Math.abs(left.value))[0];
  const targetMetric = selectedQuestion ? questionMse(selectedQuestion) : 0;
  const activeModel = formatModelName(modelMethod);
  const otherModels = availableModels.filter((model) => model && model !== modelMethod);

  if (!proxies.length && !evidenceQuestions.length) {
    return <section className="eev-empty"><Layers3 className="size-5" /><div><h2>No paired run diagnostics</h2><p>This record does not contain fitted coefficients or question-level target and prediction distributions. Its preserved overview and artifacts remain available in the other tabs.</p></div></section>;
  }

  return (
    <section className="eev-shell" aria-label="Linked ensemble and evidence workspace">
      <header className="eev-header">
        <div><p className="eev-eyebrow">Run instrument</p><h2>{view === "ensemble" ? <>From population to <em>ensemble.</em></> : <>Where does the <em>ensemble hold?</em></>}</h2></div>
        <div className="eev-header-controls">
          <div className="eev-model-context"><span>Visualized Stage 2 model</span><strong>{activeModel}</strong>{otherModels.length > 0 && <small>Also in record · {otherModels.map(formatModelName).join(", ")}</small>}</div>
          <div className="eev-switch" aria-label="Linked run views">
            <button type="button" aria-pressed={view === "ensemble"} disabled={!proxies.length} onClick={() => setView("ensemble")}><Users /> Ensemble</button>
            <button type="button" aria-pressed={view === "evidence"} disabled={!evidenceQuestions.length} onClick={() => setView("evidence")}><ScanLine /> Evidence</button>
          </div>
        </div>
      </header>

      <div className="eev-workspace">
        <div className="eev-stage">
          {view === "ensemble" ? (
            <>
              <div className="eev-stage-meta"><span>Formation · complete Stage 1 endowment budget</span><span>Colour · entropy-summary template family</span></div>
              <div className="eev-ensemble-summary" aria-label="Stage 1 population and Stage 2 selection summary">
                <div><strong>{proxies.length}</strong><span>generated proxies</span></div>
                <ArrowRight />
                <div><strong>{selectedTermCount}</strong><span>{referenceRespondent ? `${selectedProxies.length} proxies + reference` : "selected into ensemble"}</span></div>
                <div><strong>{antiTypes.length}</strong><span>negative anti-types</span></div>
                <div><strong>{proxies.length ? ((selectedProxies.length / proxies.length) * 100).toFixed(1) : "0.0"}%</strong><span>basis retained</span></div>
              </div>
              <div className="eev-formation-legend" aria-label="Ensemble formation legend">
                <span><i className="eev-legend-agent eev-legend-unselected" /> Generated, not selected</span>
                <span><i className="eev-legend-agent eev-legend-selected" /> Positive coefficient</span>
                <span><i className="eev-legend-agent eev-legend-anti" /> Negative coefficient · anti-type</span>
                <span>Selected size · |coefficient|</span>
              </div>
              <div className="eev-template-key" aria-label="Template family colours">{templateCounts.map(({ family, count }) => <span key={family}><i style={{ background: modeFamilies[family] }} />{family}<small>{count}</small></span>)}</div>
              <div className="eev-formation" aria-label="Square formation of the complete generated proxy population">
                <div className="eev-square-grid" style={{ "--eev-columns": formationColumns } as CSSProperties}>
                  {proxies.map((proxy, index) => {
                    const style = { "--eev-scale": proxy.scale, "--eev-agent-color": modeFamilies[proxy.templateFamily], "--spot": index } as CSSProperties;
                    const state = proxy.antiType ? "anti-type" : proxy.selected ? "selected" : "unselected";
                    return <button key={proxy.id} type="button" className={`eev-pop-agent ${proxy.selected ? "eev-pop-agent-selected" : ""} ${proxy.antiType ? "eev-pop-agent-anti" : ""}`} style={style} aria-pressed={selectedProxy?.id === proxy.id} title={`${proxy.id} · ${proxy.templateFamily} · ${state} · ${proxy.coefficient.toFixed(4)}`} aria-label={`${proxy.id}, ${proxy.templateFamily} template, ${state}, fitted coefficient ${proxy.coefficient.toFixed(4)}, mode ${proxy.mode}`} onClick={() => setSelectedProxyId(proxy.id)}><span className="eev-pop-person" aria-hidden="true" /></button>;
                  })}
                </div>
                {referenceRespondent && <aside className="eev-reference-respondent" aria-label={`Reference respondent model anchor, coefficient ${referenceRespondent.coefficient.toFixed(4)}`}><i className="eev-reference-person" aria-hidden="true" /><small>Model anchor</small><strong>Reference respondent</strong><code>{referenceRespondent.coefficient >= 0 ? "+" : ""}{referenceRespondent.coefficient.toFixed(4)}</code><p>Separate from the Stage 1 proxy budget</p></aside>}
              </div>
              <div className="eev-stage-foot"><span>All {proxies.length} generated proxies shown</span><span>Formation is categorical, not a map of psychological or response similarity.</span></div>
            </>
          ) : selectedQuestion ? (
            <>
              <div className="eev-stage-meta"><span>{evidenceQuestions.some((question) => question.split === "test") ? "Held-out test evidence" : "Available question diagnostics"}</span><span>Cells · prediction − target</span></div>
              <div className="eev-evidence">
                <section className="eev-question-detail">
                  <div className="eev-question-heading"><div><span>{selectedQuestion.qid} · {selectedQuestion.split}</span><h3>{selectedQuestion.question}</h3></div><strong>{targetMetric.toFixed(4)}<small> question MSE</small></strong></div>
                  <div className="eev-distribution-row"><span>Target</span><div className="eev-stacked">{selectedQuestion.option_order.map((option, index) => <i key={option} className={`eev-segment eev-segment-${index % 6}`} style={{ width: `${Math.max(0, selectedQuestion.target[option] || 0) * 100}%` }} />)}</div><span>100%</span></div>
                  <div className="eev-distribution-row"><span>Prediction</span><div className="eev-stacked">{selectedQuestion.option_order.map((option, index) => <i key={option} className={`eev-segment eev-segment-${index % 6}`} style={{ width: `${Math.max(0, selectedQuestion.prediction[option] || 0) * 100}%` }} />)}</div><span>100%</span></div>
                  <div className="eev-option-key">{selectedQuestion.option_order.slice(0, 6).map((option, index) => <span key={option}><i className={`eev-key eev-segment-${index % 6}`} />{option}</span>)}</div>
                </section>
                <section className="eev-residuals" aria-label="Question-level residuals">
                  <div className="eev-residual-head"><span>Question</span><span>Option-level residual</span><span>Error</span></div>
                  {visibleQuestions.map((question) => {
                    const error = questionMse(question);
                    return <button key={question.qid} type="button" className="eev-residual-row" aria-pressed={question.qid === selectedQuestion.qid} onClick={() => setSelectedQid(question.qid)}><span><strong>{question.qid}</strong><small>{question.question}</small></span><span className="eev-residual-cells" style={{ gridTemplateColumns: `repeat(${Math.max(1, Math.min(6, question.option_order.length))}, minmax(0, 1fr))` }}>{question.option_order.slice(0, 6).map((option) => { const residual = (question.prediction[option] || 0) - (question.target[option] || 0); return <i key={option} className={residual >= 0 ? "eev-residual-positive" : "eev-residual-negative"} style={{ opacity: Math.max(.2, Math.min(1, Math.abs(residual) * 12)) }} />; })}</span><span>{error.toFixed(4)}</span></button>;
                  })}
                </section>
              </div>
              <div className="eev-stage-foot"><span><i className="eev-residual-negative" /> Under-predicted <i className="eev-residual-positive" /> Over-predicted</span><span>Direct comparison of stored target and prediction marginals.</span></div>
            </>
          ) : null}
        </div>

        <aside className="eev-inspector" aria-live="polite">
          {view === "ensemble" && selectedProxy ? (
            <>
              <div className="eev-inspector-head"><p>Inspected proxy · {selectedProxy.id}</p><h3>{selectedProxy.mode.replaceAll("_", " ").replaceAll("+", " + ")}</h3><span className={selectedProxy.antiType ? "eev-anti-label" : ""}>{selectedProxy.antiType ? "anti-type · negative coefficient" : selectedProxy.selected ? "selected response law" : "generated · not selected"}</span></div>
              <div className={`eev-inspector-measure ${selectedProxy.antiType ? "eev-inspector-measure-anti" : ""}`}><span><Weight /> Stage 2 coefficient</span><strong>{selectedProxy.coefficient.toFixed(4)}</strong><div className="eev-measure-track"><i style={{ width: `${selectedProxy.selected ? Math.max(4, selectedProxy.magnitude / Math.max(...proxies.map((proxy) => proxy.magnitude), 1e-9) * 100) : 0}%` }} /></div>{selectedProxy.selected ? <small>{selectedProxy.antiType ? "Subtracts this response law from the fitted combination." : "Adds this response law to the fitted combination."}</small> : <small>This Stage 1 candidate received no fitted weight.</small>}</div>
              {selectedProxy.generationStep !== "not recorded" && <div className="eev-inspector-section"><span>Generation step</span><p>{selectedProxy.generationStep}</p></div>}
              <div className="eev-inspector-section"><span>Endowment</span><p>{selectedProxy.text}</p></div>
              {selectedProxy.attributes.length > 0 && <div className="eev-inspector-section"><span>Control attributes</span><div className="eev-attributes">{selectedProxy.attributes.slice(0, 14).map((attribute) => <i key={attribute}>{attribute}</i>)}</div></div>}
              <button type="button" className="eev-link-action" disabled={!evidenceQuestions.length} onClick={() => setView("evidence")}>Inspect held-out evidence <ArrowRight /></button>
            </>
          ) : selectedQuestion ? (
            <>
              <div className="eev-inspector-head"><p>Selected question · {selectedQuestion.split}</p><h3>{selectedQuestion.qid}</h3><span>{selectedQuestion.option_order.length} response options</span></div>
              <div className="eev-inspector-measure"><span><ScanLine /> Question MSE</span><strong>{targetMetric.toFixed(4)}</strong><div className="eev-measure-track"><i style={{ width: `${Math.max(4, Math.min(100, targetMetric * 1600))}%` }} /></div></div>
              <div className="eev-inspector-section"><span>Largest residual</span><p className="eev-large-residual">{largestResidual ? `${largestResidual.value >= 0 ? "+" : ""}${(largestResidual.value * 100).toFixed(1)} pp` : "—"}</p><small>{largestResidual?.option || "No option-level residual available"}</small></div>
              <div className="eev-inspector-section"><span>Interpretation</span><p>The residual is evidence about this basis and backend under the current survey instrument. It is not a recovered psychological distance.</p></div>
              <button type="button" className="eev-link-action" disabled={!proxies.length} onClick={() => setView("ensemble")}>Inspect fitted basis <ArrowRight /></button>
            </>
          ) : null}
        </aside>
      </div>
    </section>
  );
}

function HealthPanel({ health }: { health: RunHealth }) {
  const tone = { ok: "bg-emerald-500", info: "bg-sky-500", warning: "bg-amber-500" } as const;
  const order = { warning: 0, info: 1, ok: 2 } as const;
  const checks = [...health.checks].sort((left, right) => order[left.level] - order[right.level]);
  return <section className="surface-card overflow-hidden">
    <div className="flex flex-col gap-2 border-b px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6"><div><div className="flex items-center gap-2"><Activity className="size-4 text-primary" /><h2 className="section-title">Run health</h2></div><p className="section-copy">Read from this run's own records: solver, penalty selection, coverage, rescaling and answers.</p></div><Badge variant="outline" className={health.status === "healthy" ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-800"}>{health.status === "healthy" ? "No warnings" : `${health.warningCount} ${health.warningCount === 1 ? "warning" : "warnings"}`}</Badge></div>
    <div className="grid gap-0 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
      <ul className="divide-y">{checks.length ? checks.map((check) => <li key={`${check.label}-${check.detail}`} className="flex gap-3 px-5 py-3.5 sm:px-6"><span className={`mt-1.5 size-2 shrink-0 rounded-full ${tone[check.level]}`} aria-label={check.level} /><span className="min-w-0"><strong className="block text-sm">{check.label}</strong><span className="text-xs leading-5 text-muted-foreground">{check.detail}</span></span></li>) : <li className="px-6 py-4 text-sm text-muted-foreground">This record has no fitting or scoring diagnostics to check.</li>}</ul>
      <dl className="grid content-start gap-x-4 gap-y-3 border-t bg-muted/20 p-5 text-sm sm:grid-cols-2 lg:border-l lg:border-t-0">{health.facts.map((fact) => <div key={fact.label} className="min-w-0"><dt className="text-xs text-muted-foreground">{fact.label}</dt><dd className="mt-0.5 truncate font-mono text-xs" title={fact.hint || fact.value}>{fact.value}</dd></div>)}</dl>
    </div>
  </section>;
}

function AggregateRow({ item, onOpen }: { item: AggregateSummary; onOpen: () => void }) {
  return <button onClick={onOpen} className="grid w-full gap-2 border-b px-5 py-4 text-left transition-colors last:border-0 hover:bg-muted/45 lg:grid-cols-[minmax(260px,1.6fr)_90px_80px_80px_minmax(160px,1fr)_100px_24px] lg:items-center lg:gap-3"><span className="min-w-0"><strong className="block truncate text-sm">{item.name}</strong><span className="block truncate font-mono text-[11px] text-muted-foreground">{item.path}</span></span><Badge variant="outline" className={`w-fit ${item.status === "valid" ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-800"}`}>{item.status}</Badge><span className="font-mono text-sm">{formatNumber(item.includedRunCount ?? item.runCount)}</span><span className="font-mono text-sm">{formatNumber(item.groupCount)}</span><span className="truncate text-xs text-muted-foreground">{item.groupKeys.join(" × ") || "—"}</span><span className="text-xs text-muted-foreground">{formatDate(item.modifiedAt)}</span><ChevronRight className="hidden size-4 text-muted-foreground lg:block" /></button>;
}

function AggregateResult({ detail, catalog, onBack }: { detail: AggregateDetail; catalog: MetricInfo[]; onBack: () => void }) {
  const summary = detail.summary;
  const [filter, setFilter] = useState("");
  const metrics = catalog.map((item) => item.id).filter((id) => summary.metrics.includes(id)).concat(summary.metrics.filter((id) => !catalog.some((item) => item.id === id)));
  const groups = useMemo(() => {
    const table = new Map<string, { key: (string | number | null)[]; cells: Record<string, AggregateRow> }>();
    detail.rows.forEach((row) => {
      const key = summary.groupKeys.map((name) => row[name] ?? null);
      const id = JSON.stringify(key);
      const group = table.get(id) || { key, cells: {} };
      group.cells[row.metric] = row;
      table.set(id, group);
    });
    const needle = filter.trim().toLowerCase();
    return [...table.values()].filter((group) => !needle || group.key.join(" ").toLowerCase().includes(needle));
  }, [detail.rows, summary.groupKeys, filter]);
  const downloads = detail.files.filter((name) => /\.(csv|json|ya?ml|md|tex)$/i.test(name));
  return <>
    <Button variant="ghost" className="mb-4 -ml-3 text-muted-foreground" onClick={onBack}><ArrowLeft /> Results library</Button>
    <Header eyebrow="Pooled summary · p2p aggregate" title={summary.name} copy={`${formatNumber(summary.includedRunCount ?? summary.runCount)} runs summarised into ${formatNumber(summary.groupCount)} groups by ${summary.groupKeys.join(" × ") || "—"}. Centre: ${summary.center || "mean"}; spread: ${summary.spread || "sample SD"}.`} />
    <section className="surface-card mb-5 p-5 sm:p-6"><div className="flex flex-wrap items-center gap-2"><Badge variant="outline" className={summary.status === "valid" ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-800"}>{summary.status}</Badge>{Object.entries(detail.checks).map(([name, status]) => <Badge key={name} variant="outline" className={`font-normal ${status === "pass" ? "" : "border-amber-200 bg-amber-50 text-amber-800"}`}>{name.replaceAll("_", " ")}: {status}</Badge>)}{detail.missingRows.length > 0 && <Badge variant="outline" className="border-amber-200 bg-amber-50 text-amber-800">{detail.missingRows.length} missing rows ({detail.missingPolicy})</Badge>}</div>{summary.digest && <p className="mt-3 truncate font-mono text-[11px] text-muted-foreground" title={summary.digest}>aggregation digest {summary.digest}</p>}<div className="mt-4 flex flex-wrap gap-2">{downloads.map((name) => <a key={name} href={fileUrl(`${summary.path}/${name}`)} download className="inline-flex h-8 items-center gap-2 rounded-md border bg-background px-3 text-sm font-medium shadow-xs transition-colors hover:bg-muted"><Download className="size-4" /> {name}</a>)}</div></section>
    <section className="surface-card overflow-hidden"><div className="flex flex-col gap-3 border-b px-5 py-4 sm:flex-row sm:items-center sm:justify-between"><div><h2 className="section-title">Recorded summary</h2><p className="section-copy">Values exactly as p2p aggregate wrote them: centre ± spread, with the number of runs per cell.</p></div><div className="relative w-full sm:w-72"><Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input aria-label="Filter groups" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter groups…" className="bg-card pl-9" /></div></div>
      <div className="overflow-x-auto"><table className="w-full min-w-[720px] text-sm"><thead className="bg-muted/35 text-left text-xs uppercase tracking-[.06em] text-muted-foreground"><tr>{summary.groupKeys.map((name) => <th key={name} className="px-4 py-3">{name}</th>)}{metrics.map((id) => { const info = metricInfo(catalog, id); return <th key={id} className="px-4 py-3 text-right" title={info.description}>{info.short} {info.direction === "lower" ? "↓" : "↑"}</th>; })}</tr></thead><tbody>{groups.map((group) => <tr key={JSON.stringify(group.key)} className="border-t">{group.key.map((value, index) => <td key={summary.groupKeys[index]} className="px-4 py-2.5 font-mono text-xs">{value == null ? "—" : String(value)}</td>)}{metrics.map((id) => { const cell = group.cells[id]; return <td key={id} className="px-4 py-2.5 text-right font-mono text-xs">{cell ? <>{formatNumber(cell.center ?? undefined, 4)}<span className="text-muted-foreground"> ± {formatNumber(cell.spread ?? undefined, 4)}</span><span className="ml-1 text-[10px] text-muted-foreground">n={cell.n}</span></> : "—"}</td>; })}</tr>)}</tbody></table></div>
    </section>
  </>;
}

function CanonicalRun({ run, detail, catalog, annotation, onAnnotation, onBack }: { run: Run; detail: RunDetail; catalog: MetricInfo[]; annotation?: Annotation; onAnnotation: (annotation: Annotation) => void; onBack: () => void }) {
  const questions = detail.questions || [];
  const hasPairedWorkspace = questions.length > 0 || Object.keys(detail.model?.coefficients || {}).length > 0;
  const [tab, setTab] = useState<"workspace" | "overview" | "predictions" | "entropy" | "weights" | "archive">(hasPairedWorkspace ? "workspace" : "overview");
  useEffect(() => { setTab(hasPairedWorkspace ? "workspace" : "overview"); }, [run.path, hasPairedWorkspace]);
  const coveredQuestions = Math.max(
    questions.length,
    run.questionCount,
    ...run.metrics.map((metric) => metric.qidsUsed || 0),
  );
  const lineage = run.origin === "legacy" ? "Earlier-format run (read-only)" : "Current pipeline";
  const generation = detail.archive || detail;
  const modelMethod = detail.model?.method || run.method;
  const availableModels = Array.from(new Set([
    modelMethod,
    ...(detail.metrics?.metrics || []).map((metric) => metric.model),
    ...(detail.archive?.summary.models || []),
  ].filter((model): model is string => Boolean(model))));
  const tabs = [
    { id: "workspace" as const, label: "Ensemble ↔ Evidence" },
    { id: "overview" as const, label: "Run record" },
    ...(run.capabilities?.predictions ? [{ id: "predictions" as const, label: "Prediction explorer", count: questions.length }] : []),
    ...(run.capabilities?.generation ? [{ id: "entropy" as const, label: "Entropy explorer" }] : []),
    ...(run.capabilities?.weights ? [{ id: "weights" as const, label: "Proxy weights" }] : []),
    ...(detail.archive ? [{ id: "archive" as const, label: "Original report files", count: detail.archive.documents.length + detail.archive.images.length }] : []),
  ];
  const testMetrics = Object.fromEntries(run.metrics.filter((metric) => metric.split === "test").map((metric) => [metric.metric, metric.value]));
  // Metrics are scored on the held-out test split only; say how many questions that is.
  const testScored = Math.max(0, ...run.metrics.filter((metric) => metric.split === "test").map((metric) => metric.qidsUsed || 0));
  const partialCoverage = run.metrics.find((metric) => metric.qidsRequested != null && (metric.qidsUsed || 0) < metric.qidsRequested);
  const downloads = ["run.json", "metrics.json", ...(run.capabilities?.usage ? ["usage.json"] : []), ...(run.capabilities?.folds ? ["folds.json"] : []), ...(run.hasDiagnostics ? ["diagnostics.json"] : [])];
  return <>
    <Button variant="ghost" className="mb-4 -ml-3 text-muted-foreground" onClick={onBack}><ArrowLeft /> Results library</Button>
    <Header eyebrow={`P2P run · ${lineage}`} title={annotation?.label || humanizeRunId(run.runId)} copy={`Completed ${formatDate(run.modifiedAt)} · seed ${run.seed ?? "—"} · ${run.method || "model unavailable"}`} />
    <p className="run-technical-id" title={run.path}><strong>{run.runId}</strong><span>{run.path}</span></p>
    <AnnotationEditor path={run.path} annotation={annotation} onSaved={onAnnotation} />
    <Tabs value={tab} onChange={setTab} items={tabs} sticky />
    {tab === "workspace" ? <EnsembleEvidenceView questions={questions} coefficients={detail.model?.coefficients} endowments={generation.endowmentRows || []} modelMethod={modelMethod} availableModels={availableModels} /> : tab === "overview" ? <div className="space-y-5">
      <section className="surface-card border-l-4 border-l-primary p-5 sm:p-6"><p className="page-eyebrow mb-2">Result at a glance</p><p className="text-sm leading-6">{testScored ? <>This record scores <strong>{testScored} held-out test {testScored === 1 ? "question" : "questions"}</strong>{coveredQuestions > testScored ? <> (predictions exist for {coveredQuestions} questions in all)</> : null}</> : <>This record covers <strong>{coveredQuestions || "an unavailable number of"} questions</strong></>}{testMetrics.mse != null ? <> with test MSE <strong className="font-mono">{formatNumber(testMetrics.mse, 4)}</strong></> : null}{testMetrics.one_minus_jsd != null ? <> and test 1 − JSD <strong className="font-mono">{formatNumber(testMetrics.one_minus_jsd, 4)}</strong></> : null}. Metric direction and prediction policy are shown below; Studio does not infer statistical significance from a single run.</p>{partialCoverage && <p className="mt-3 flex gap-2 text-sm text-amber-700"><TriangleAlert className="mt-0.5 size-4 shrink-0" />At least one metric used {partialCoverage.qidsUsed} of {partialCoverage.qidsRequested} requested questions.</p>}{detail.health && detail.health.warningCount > 0 && <p className="mt-3 flex gap-2 text-sm text-amber-700"><TriangleAlert className="mt-0.5 size-4 shrink-0" />Run health lists {detail.health.warningCount} {detail.health.warningCount === 1 ? "warning" : "warnings"} below.</p>}</section>
      {detail.health && <HealthPanel health={detail.health} />}
      {run.origin === "legacy" && <section className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-3"><div className="flex flex-wrap items-center gap-2"><Badge>Earlier format</Badge><span className="text-sm font-medium">Read from its original files without changing them.</span></div>{detail.legacy?.notes?.map((note) => <p key={note} className="mt-2 text-xs leading-5 text-muted-foreground">{note}</p>)}</section>}
      {detail.diagnosticsAvailable === false && <section className="rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sky-900"><div className="flex flex-wrap items-center gap-2"><Badge variant="outline" className="border-sky-300 bg-white">Evaluation-only run</Badge><span className="text-sm font-medium">Canonical metrics are complete; fitting diagnostics do not apply.</span></div>{detail.notes?.map((note) => <p key={note} className="mt-2 text-xs leading-5 text-sky-800">{note}</p>)}</section>}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2"><section className="surface-card p-5 sm:p-6"><div className="mb-5 flex items-center gap-2"><Gauge className="size-4 text-primary" /><h2 className="section-title">Evaluation metrics</h2></div><MetricBars metrics={run.metrics} catalog={catalog} /></section><section className="surface-card p-5 sm:p-6"><div className="mb-5 flex items-center gap-2"><Activity className="size-4 text-primary" /><h2 className="section-title">Run footprint</h2></div><div className="grid grid-cols-2 gap-3">{[["Input tokens", formatNumber(run.usage.input_tokens)], ["Output tokens", formatNumber(run.usage.output_tokens)], ["Cost", formatMoney(run.usage.cost_usd)], [detail.diagnosticsAvailable === false ? "Questions covered" : "Questions", formatNumber(coveredQuestions)]].map(([label,value]) => <div key={label} className="rounded-xl border bg-background p-4"><p className="text-xs text-muted-foreground">{label}</p><p className="mt-1 text-xl font-semibold">{value}</p></div>)}</div></section></div>
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2"><section className="surface-card p-5 sm:p-6"><div className="mb-4"><h2 className="section-title">Provenance and compatibility</h2><p className="section-copy">Where this record came from and which contract it satisfies.</p></div><dl className="space-y-3 text-sm"><div className="flex justify-between gap-4"><dt className="text-muted-foreground">Source format</dt><dd className="font-mono text-xs">{run.sourceFormat || (run.origin === "legacy" ? "earlier format" : "canonical")}</dd></div><div className="flex justify-between gap-4"><dt className="text-muted-foreground">Schema version</dt><dd className="font-mono">{run.sourceVersion ?? "—"}</dd></div><div className="flex justify-between gap-4"><dt className="text-muted-foreground">Lineage</dt><dd>{run.provenance?.label || lineage}</dd></div>{detail.job && <><div className="flex justify-between gap-4"><dt className="text-muted-foreground">Studio job</dt><dd className="font-mono text-xs">{detail.job.id} · {detail.job.status}</dd></div><div className="flex justify-between gap-4"><dt className="text-muted-foreground">Manifest</dt><dd className="max-w-[70%] truncate font-mono text-xs" title={detail.job.config}>{detail.job.config}</dd></div></>}</dl>{run.compatibilityWarnings?.map((warning) => <p key={warning} className="mt-3 text-xs text-amber-700">{warning}</p>)}</section><section className="surface-card p-5 sm:p-6"><div className="mb-4"><h2 className="section-title">Canonical artifacts</h2><p className="section-copy">Download source records without modifying the run directory.</p></div><div className="flex flex-wrap gap-2">{downloads.map((name) => <a key={name} href={fileUrl(`${run.path}/${name}`)} download className="inline-flex h-8 items-center gap-2 rounded-md border bg-background px-3 text-sm font-medium shadow-xs transition-colors hover:bg-muted"><Download className="size-4" /> {name}</a>)}</div></section></div>
    </div> : tab === "predictions" ? <PredictionExplorer questions={questions} /> : tab === "entropy" ? <div className="space-y-5"><ModeEntropyAnalysis entropyRows={generation.modeEntropyRows || []} summaryRows={generation.modeSummaryRows || []} selectionRows={generation.modeRows || []} selectionSource={generation.modeSelectionSource} preferredModel={detail.model?.method} /><EntropyExplorer questions={questions} /></div> : tab === "weights" ? <Coefficients coefficients={detail.model?.coefficients} endowments={generation.endowmentRows || []} /> : detail.archive ? <RunArchive detail={detail.archive} /> : null}
  </>;
}

function inlineMarkdown(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean).map((part, index) => part.startsWith("`") ? <code key={index} className="rounded bg-muted px-1 py-0.5">{part.slice(1,-1)}</code> : part.startsWith("**") ? <strong key={index}>{part.slice(2,-2)}</strong> : part);
}

function MarkdownDocument({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const nodes: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line || line.startsWith("<img")) { index += 1; continue; }
    if (line.startsWith("```")) {
      const code: string[] = []; index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) { code.push(lines[index]); index += 1; }
      nodes.push(<pre key={`code-${index}`} className="my-4 max-h-80 overflow-auto rounded-xl reading-surface p-4">{code.join("\n")}</pre>); index += 1; continue;
    }
    if (line.startsWith("|")) {
      const table: string[][] = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) { table.push(lines[index].trim().slice(1,-1).split("|").map((cell) => cell.trim())); index += 1; }
      const rows = table.filter((_, rowIndex) => rowIndex !== 1 || !table[rowIndex].every((cell) => /^:?-+:?$/.test(cell)));
      if (rows.length) nodes.push(<div key={`table-${index}`} className="my-4 overflow-x-auto"><table className="w-full min-w-[520px] text-left text-xs"><thead><tr>{rows[0].map((cell, cellIndex) => <th key={cellIndex} className="border-b bg-muted/50 px-3 py-2 font-semibold">{inlineMarkdown(cell)}</th>)}</tr></thead><tbody>{rows.slice(1,31).map((row,rowIndex) => <tr key={rowIndex} className="border-b">{row.map((cell,cellIndex) => <td key={cellIndex} className="px-3 py-2 align-top">{inlineMarkdown(cell)}</td>)}</tr>)}</tbody></table>{rows.length > 31 && <p className="mt-2 text-xs text-muted-foreground">Showing 30 of {rows.length - 1} rows.</p>}</div>); continue;
    }
    if (/^#{1,4}\s/.test(line)) { const level = line.match(/^#+/)?.[0].length || 2; const text = line.replace(/^#+\s*/, ""); nodes.push(level <= 2 ? <h2 key={index} className="mb-2 mt-7 text-lg font-semibold">{inlineMarkdown(text)}</h2> : <h3 key={index} className="mb-2 mt-5 font-semibold">{inlineMarkdown(text)}</h3>); index += 1; continue; }
    if (line.startsWith("- ")) { const items: string[] = []; while (index < lines.length && lines[index].trim().startsWith("- ")) { items.push(lines[index].trim().slice(2)); index += 1; } nodes.push(<ul key={`list-${index}`} className="my-3 list-disc space-y-1 pl-5 text-sm leading-6 text-muted-foreground">{items.map((item,itemIndex) => <li key={itemIndex}>{inlineMarkdown(item)}</li>)}</ul>); continue; }
    const paragraph: string[] = [line]; index += 1;
    while (index < lines.length && lines[index].trim() && !/^(#|\||```|- |<img|\*\*)/.test(lines[index].trim())) { paragraph.push(lines[index].trim()); index += 1; }
    nodes.push(<p key={`p-${index}`} className="my-2 text-sm leading-6 text-muted-foreground">{inlineMarkdown(paragraph.join(" ").replace(/\s{2,}/g," "))}</p>);
  }
  return <div>{nodes}</div>;
}

function ImageGallery({ images, empty }: { images: ReportDetail["images"]; empty: string }) {
  if (!images.length) return <div className="surface-card flex min-h-48 flex-col items-center justify-center text-center"><ImageIcon className="mb-3 size-5 text-muted-foreground" /><p className="text-sm text-muted-foreground">{empty}</p></div>;
  return <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">{images.map((image) => <figure key={image.path} className="surface-card overflow-hidden"><a href={artifactUrl(image.path)} target="_blank" rel="noreferrer" className="block bg-white p-3"><img src={artifactUrl(image.path)} alt={image.name.replaceAll("_"," ")} loading="lazy" className="h-auto max-h-[520px] w-full object-contain" /></a><figcaption className="flex items-center justify-between gap-3 border-t px-4 py-3 text-xs"><span className="truncate font-medium">{image.name.replace(/\.[^.]+$/,"").replaceAll("_"," ")}</span><span className="text-muted-foreground">{formatBytes(image.size)}</span></figcaption></figure>)}</div>;
}

function RunArchive({ detail }: { detail: ReportDetail }) {
  const [documentName, setDocumentName] = useState(detail.documents[0]?.name || "");
  const document = detail.documents.find((item) => item.name === documentName) || detail.documents[0];
  return <div className="space-y-5"><section className="surface-card p-5 sm:p-6"><h2 className="section-title">Original report files</h2><p className="section-copy">The figures and Markdown reports written with this run, shown as they were. The prediction, entropy and weight tabs use the shared interactive views.</p></section>{detail.images.length > 0 && <ImageGallery images={detail.images} empty="No archived figures were found." />}{detail.documents.length > 0 && <div className="grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)]"><aside className="surface-card self-start overflow-hidden">{detail.documents.map((item) => <button key={item.name} onClick={() => setDocumentName(item.name)} className={`flex w-full items-center gap-3 border-b px-4 py-3 text-left text-sm last:border-0 ${document?.name === item.name ? "bg-primary/8 text-primary" : "hover:bg-muted"}`}><FileText className="size-4 shrink-0" /><span className="min-w-0"><strong className="block truncate">{item.title}</strong><span className="block truncate font-mono text-[11px] text-muted-foreground">{item.name}</span></span></button>)}</aside><article className="surface-card min-w-0 p-5 sm:p-7">{document && <MarkdownDocument content={document.content} />}</article></div>}</div>;
}

function ReportMetrics({ report }: { report: Report }) {
  if (!report.metrics.length) return <p className="text-sm text-muted-foreground">No summary_aggregate.csv metrics were found.</p>;
  const fields = [{key:"testMse",label:"Test MSE"},{key:"testOneMinusJsd",label:"1 − JSD"},{key:"testEmd",label:"EMD"},{key:"testAccuracy",label:"Accuracy"}] as const;
  return <div className="space-y-6">{report.metrics.map((row) => <div key={row.model} className="border-b pb-5 last:border-0"><div className="mb-3 flex items-center justify-between"><h3 className="font-semibold">{row.model}</h3>{row.selected != null && <Badge variant="outline">{formatNumber(row.selected)} selected</Badge>}</div><div className="grid gap-3 sm:grid-cols-2">{fields.map((field) => <div key={field.key}><div className="mb-1 flex justify-between text-xs"><span className="text-muted-foreground">{field.label}</span><span className="font-mono">{formatNumber(row[field.key],4)}</span></div><div className="h-2 rounded-full bg-muted"><div className="viz-metric-fill h-full rounded-full" style={{width:`${Math.max(2,Math.min(100,(row[field.key] || 0)*100))}%`}} /></div></div>)}</div></div>)}</div>;
}

function EntropyTrajectory({ rows }: { rows: Record<string,string>[] }) {
  const points = rows.map((row) => ({x:Number(row.step),y:Number(row.entropy)})).filter((point) => Number.isFinite(point.x)&&Number.isFinite(point.y));
  if (points.length < 2) return null;
  const minY = Math.min(...points.map((point) => point.y)); const maxY = Math.max(...points.map((point) => point.y)); const span = Math.max(maxY-minY,.01);
  const maxX = Math.max(...points.map((item) => item.x),1);
  const xTicks = generationStepTicks(maxX);
  const x = (value: number) => 58 + value/maxX*512;
  const y = (value: number) => 210-(value-minY)/span*170;
  const path = points.map((point,index) => `${index ? "L" : "M"} ${x(point.x)} ${y(point.y)}`).join(" ");
  return <section className="surface-card p-5 sm:p-6"><h2 className="section-title">Aggregate entropy trajectory</h2><p className="section-copy">How response diversity changed across generation updates.</p><div className="mx-auto mt-3 w-full max-w-[680px]"><svg viewBox="0 0 600 260" className="block h-auto w-full" role="img" aria-label="Aggregate entropy by generation step"><rect x="58" y="20" width="512" height="190" fill="var(--background)" stroke="var(--border)" />{xTicks.map((tick) => <g key={tick}><line x1={x(tick)} y1="210" x2={x(tick)} y2="216" stroke="var(--muted-foreground)" /><text x={x(tick)} y="230" textAnchor={tick === 0 ? "start" : tick === maxX ? "end" : "middle"} fill="var(--muted-foreground)" fontSize="11">{tick}</text></g>)}<path d={path} fill="none" stroke="var(--viz-prediction)" strokeWidth="3" />{points.map((point) => <circle key={point.x} cx={x(point.x)} cy={y(point.y)} r="4" fill="var(--viz-prediction)"><title>Step {point.x}: {point.y.toFixed(3)}</title></circle>)}<text x="314" y="254" textAnchor="middle" fill="var(--foreground)" fontSize="12">Generation step</text><text x="16" y="115" transform="rotate(-90 16 115)" textAnchor="middle" fill="var(--foreground)" fontSize="12">Normalized entropy</text><text x="50" y="43" textAnchor="end" fill="var(--muted-foreground)" fontSize="12">{maxY.toFixed(2)}</text><text x="50" y="212" textAnchor="end" fill="var(--muted-foreground)" fontSize="12">{minY.toFixed(2)}</text></svg></div></section>;
}

function generationStepTicks(maxStep: number, maxTicks = 6) {
  const maximum = Math.max(0, Math.ceil(maxStep));
  if (maximum <= maxTicks - 1) return Array.from({ length: maximum + 1 }, (_, index) => index);
  const roughStep = maximum / (maxTicks - 1);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const niceStep = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
  const ticks = Array.from({ length: Math.floor(maximum / niceStep) + 1 }, (_, index) => index * niceStep);
  if (ticks.at(-1) !== maximum) ticks.push(maximum);
  return ticks;
}

function csvNumber(value?: string) {
  if (value == null || value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

type ModeFamily = keyof typeof modeFamilies;
type ModeSort = "entropy" | "generated" | "selected" | "weight";

function ModeEntropyAnalysis({
  entropyRows,
  summaryRows,
  selectionRows,
  selectionSource,
  preferredModel,
}: {
  entropyRows: Record<string,string>[];
  summaryRows: Record<string,string>[];
  selectionRows: Record<string,string>[];
  selectionSource?: "summary_by_mode" | "derived_coefficients" | "unavailable";
  preferredModel?: string;
}) {
  const models = useMemo(() => Array.from(new Set(selectionRows.map((row) => row.model).filter(Boolean))), [selectionRows]);
  const [model, setModel] = useState("");
  const defaultModel = preferredModel && models.includes(preferredModel) ? preferredModel : models.includes("lasso") ? "lasso" : models[0] || "";
  const activeModel = models.includes(model) ? model : defaultModel;
  const selectionAvailable = selectionRows.length > 0;
  const [sort, setSort] = useState<ModeSort>("entropy");
  const activeSort: ModeSort = !selectionAvailable && (sort === "selected" || sort === "weight") ? "entropy" : sort;
  const modes = useMemo(() => {
    const entropyNames = Object.keys(entropyRows[0] || {}).filter((name) => name !== "step");
    const summaryByMode = new Map(summaryRows.filter((row) => row.mode).map((row) => [row.mode, row]));
    const selectionByMode = new Map(selectionRows.filter((row) => row.mode && (!activeModel || row.model === activeModel)).map((row) => [row.mode, row]));
    const names = Array.from(new Set([...entropyNames, ...summaryByMode.keys(), ...selectionByMode.keys()]));
    return names.map((name) => {
      const trajectory = entropyRows.flatMap((row, index) => {
        const value = csvNumber(row[name]);
        if (value == null) return [];
        return [{ step: csvNumber(row.step) ?? index, value }];
      });
      const summary = summaryByMode.get(name);
      const selection = selectionByMode.get(name);
      const attributes = (summary?.attributes || "").split(/;\s*/).map((item) => item.trim()).filter(Boolean);
      return {
        mode: name,
        family: modeFamily(name),
        trajectory,
        finalEntropy: trajectory.at(-1)?.value,
        entropyChange: trajectory.length > 1 ? trajectory.at(-1)!.value - trajectory[0].value : 0,
        generated: csvNumber(summary?.n) || 0,
        attributes,
        selected: csvNumber(selection?.num_selected) || 0,
        totalWeight: csvNumber(selection?.total_abs_weight) ?? Math.abs(csvNumber(selection?.total_weight) || 0),
        fractionSelected: csvNumber(selection?.fraction_selected) || 0,
        averageWeight: csvNumber(selection?.avg_weight) || 0,
      };
    });
  }, [entropyRows, summaryRows, selectionRows, activeModel]);
  const ranked = useMemo(() => [...modes].sort((left, right) => {
    const values: Record<ModeSort, [number, number]> = {
      entropy: [left.finalEntropy ?? -1, right.finalEntropy ?? -1],
      generated: [left.generated, right.generated],
      selected: [left.selected, right.selected],
      weight: [left.totalWeight, right.totalWeight],
    };
    return values[activeSort][1] - values[activeSort][0] || left.mode.localeCompare(right.mode);
  }), [modes, activeSort]);
  const [selectedMode, setSelectedMode] = useState("");
  const selected = modes.find((item) => item.mode === selectedMode) || ranked[0];
  if (!modes.length) return null;

  const totalGenerated = modes.reduce((sum, item) => sum + item.generated, 0);
  const totalWeight = modes.reduce((sum, item) => sum + Math.max(0, item.totalWeight), 0);
  const plotModes = modes.filter((item) => item.finalEntropy != null);
  const minEntropy = Math.min(...plotModes.map((item) => item.finalEntropy!), 0);
  const maxEntropy = Math.max(...plotModes.map((item) => item.finalEntropy!), .1);
  const entropySpan = Math.max(maxEntropy - minEntropy, .05);
  const maxSelected = Math.max(...plotModes.map((item) => item.selected), 1);
  const maxGenerated = Math.max(...plotModes.map((item) => item.generated), 1);
  const selectedTicks = Array.from(new Set([0, Math.ceil(maxSelected / 2), maxSelected])).sort((left, right) => left - right);
  const plotX = (value: number) => 64 + (value - minEntropy) / entropySpan * 490;
  const plotY = (value: number) => 242 - value / maxSelected * 194;
  const entropyRank = selected ? [...modes].sort((a,b) => (b.finalEntropy ?? -1) - (a.finalEntropy ?? -1)).findIndex((item) => item.mode === selected.mode) + 1 : 0;
  const trajectoryMax = selected ? Math.max(...selected.trajectory.map((point) => point.value), .1) : 1;
  const trajectoryStepMax = selected ? Math.max(...selected.trajectory.map((point) => point.step), 1) : 1;
  const trajectoryStepTicks = generationStepTicks(trajectoryStepMax);
  const trajectoryPath = selected?.trajectory.map((point, index) => `${index ? "L" : "M"} ${58 + point.step / trajectoryStepMax * 512} ${205 - point.value / trajectoryMax * 165}`).join(" ") || "";

  return <div className="space-y-5">
    <section className="surface-card p-5 sm:p-6">
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div><div className="flex items-center gap-2"><Sparkles className="size-4 text-primary" /><h2 className="section-title">Mode entropy and selection map</h2></div><p className="section-copy">Follow attribute-bank modes from generation to fitted proxy selection. Position shows final entropy and selected agents; point size shows generated contribution.</p>{selectionAvailable && <p className="mt-1 text-[11px] text-muted-foreground">Selection summary · {selectionSource === "derived_coefficients" ? "derived from canonical coefficients and endowments" : "stored with the run"}</p>}</div>
        {models.length > 0 && <label className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">Selection model<select value={activeModel} onChange={(event) => setModel(event.target.value)} className="mt-1.5 block h-9 min-w-40 rounded-md border bg-background px-3 text-sm font-normal normal-case tracking-normal text-foreground">{models.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>}
      </div>
      {!selectionAvailable && <div className="mb-5 flex gap-3 rounded-xl border border-amber-400/30 bg-amber-400/8 p-4 text-sm"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-500" /><div><strong>Mode-level selection is unavailable.</strong><p className="mt-1 text-xs leading-5 text-muted-foreground">Entropy and generation artifacts are present, but this record has neither a stored mode summary nor coefficients that can be joined to endowments. Studio will not interpret missing selection data as zero.</p></div></div>}
      <div className="grid gap-5 xl:grid-cols-[1.25fr_.9fr]">
        <figure className="min-w-0 rounded-xl border bg-background p-3">
          {selectionAvailable ? <svg viewBox="0 0 600 300" className="block h-auto w-full" role="img" aria-label="Mode entropy versus selected proxy count">
            <rect x="64" y="28" width="490" height="214" fill="var(--brand-paper)" stroke="var(--border)" />
            {[0,.5,1].map((tick) => <g key={`x-${tick}`}><line x1={64 + tick * 490} y1="242" x2={64 + tick * 490} y2="248" stroke="var(--muted-foreground)" /><text x={64 + tick * 490} y="266" textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"} fill="var(--muted-foreground)" fontSize="11">{(minEntropy + tick * entropySpan).toFixed(2)}</text></g>)}
            {selectedTicks.map((tick) => <g key={`y-${tick}`}><line x1="58" y1={plotY(tick)} x2="64" y2={plotY(tick)} stroke="var(--muted-foreground)" /><text x="52" y={plotY(tick) + 4} textAnchor="end" fill="var(--muted-foreground)" fontSize="11">{tick}</text></g>)}
            {plotModes.map((item) => <circle key={item.mode} onClick={() => setSelectedMode(item.mode)} cx={plotX(item.finalEntropy!)} cy={plotY(item.selected)} r={5 + Math.sqrt(item.generated / maxGenerated) * 9} fill={modeFamilies[item.family]} fillOpacity={selected?.mode === item.mode ? .95 : .68} stroke={selected?.mode === item.mode ? "var(--foreground)" : "white"} strokeWidth={selected?.mode === item.mode ? 2.5 : 1.5} className="cursor-pointer"><title>{item.mode}: entropy {item.finalEntropy?.toFixed(3)}, {item.generated} generated, {item.selected} selected</title></circle>)}
            <text x="309" y="292" textAnchor="middle" fill="var(--foreground)" fontSize="12" fontWeight="600">Final normalized entropy</text><text x="16" y="135" transform="rotate(-90 16 135)" textAnchor="middle" fill="var(--foreground)" fontSize="12" fontWeight="600">Agents selected</text>
          </svg> : <div className="flex aspect-[2/1] min-h-64 items-center justify-center rounded-lg bg-muted/20 px-8 text-center"><div><TriangleAlert className="mx-auto mb-3 size-5 text-amber-500" /><p className="text-sm font-medium">No selection map can be drawn safely.</p><p className="mt-1 max-w-md text-xs leading-5 text-muted-foreground">Open the mode ranking for the available entropy and generation summaries; selected-agent positions require fitted coefficients joined to proxy modes.</p></div></div>}
          <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-2 text-[11px] text-muted-foreground">{Object.entries(modeFamilies).map(([family,color]) => <span key={family}><i className="mr-1.5 inline-block size-2.5 rounded-full align-[-1px]" style={{backgroundColor:color}} />{family}</span>)}</figcaption>
        </figure>
        <div className="min-w-0 overflow-hidden rounded-xl border bg-background">
          <div className="border-b p-3"><p className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">Rank modes by</p><div className="mt-2 flex flex-wrap gap-1">{([ ["entropy","Entropy"], ["generated","Generated"], ["selected","Selected"], ["weight","Weight"] ] as [ModeSort,string][]).map(([id,label]) => { const disabled = !selectionAvailable && (id === "selected" || id === "weight"); return <button key={id} disabled={disabled} onClick={() => setSort(id)} className={`rounded-md px-2.5 py-1.5 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40 ${activeSort === id ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:text-foreground"}`}>{label}</button>; })}</div></div>
          <div className="grid grid-cols-[2rem_minmax(150px,1fr)_58px_58px] gap-2 border-b bg-muted/35 px-3 py-2 text-[10px] font-semibold uppercase tracking-[.06em] text-muted-foreground"><span>Rank</span><span>Mode</span><span className="text-right">Entropy</span><span className="text-right">{activeSort === "weight" ? "Weight" : activeSort === "selected" ? "Selected" : "Gen."}</span></div>
          <div className="max-h-[330px] overflow-auto">{ranked.map((item,index) => <button key={item.mode} onClick={() => setSelectedMode(item.mode)} className={`grid w-full grid-cols-[2rem_minmax(150px,1fr)_58px_58px] gap-2 border-b px-3 py-2.5 text-left text-xs last:border-0 ${selected?.mode === item.mode ? "bg-primary/8" : "hover:bg-muted/50"}`}><span className="font-mono text-muted-foreground">{index + 1}</span><span className="truncate font-medium" title={item.mode}>{item.mode}</span><span className="text-right font-mono">{item.finalEntropy?.toFixed(3) ?? "—"}</span><span className="text-right font-mono">{activeSort === "weight" ? item.totalWeight.toFixed(3) : activeSort === "selected" ? item.selected : item.generated}</span></button>)}</div>
        </div>
      </div>
    </section>
    {selected && <section className="surface-card p-5 sm:p-6">
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"><div className="min-w-0"><p className="page-eyebrow mb-2">Selected mode · entropy rank {entropyRank} of {modes.length}</p><h2 className="break-words text-lg font-semibold">{selected.mode}</h2><div className="mt-2 flex flex-wrap gap-1.5">{selected.mode.split("+").map((token) => <Badge key={token} variant="outline" style={{borderColor:modeFamilies[selected.family],color:modeFamilies[selected.family]}}>{token}</Badge>)}</div></div><Badge variant="secondary">{selected.family}</Badge></div>
      <div className="grid gap-5 xl:grid-cols-[1.2fr_.8fr]">
        <figure className="min-w-0 rounded-xl border bg-background p-3"><svg viewBox="0 0 600 250" className="block h-auto w-full" role="img" aria-label={`Entropy trajectory for ${selected.mode}`}><rect x="58" y="20" width="512" height="185" fill="var(--brand-paper)" stroke="var(--border)" />{trajectoryStepTicks.map((tick) => { const tickX = 58 + tick / trajectoryStepMax * 512; return <g key={tick}><line x1={tickX} y1="205" x2={tickX} y2="211" stroke="var(--muted-foreground)" /><text x={tickX} y="224" textAnchor={tick === 0 ? "start" : tick === trajectoryStepMax ? "end" : "middle"} fill="var(--muted-foreground)" fontSize="11">{tick}</text></g>; })}<path d={trajectoryPath} fill="none" stroke={modeFamilies[selected.family]} strokeWidth="3" />{selected.trajectory.map((point) => <circle key={`${point.step}-${point.value}`} cx={58 + point.step / trajectoryStepMax * 512} cy={205 - point.value / trajectoryMax * 165} r="4" fill={modeFamilies[selected.family]}><title>Step {point.step}: {point.value.toFixed(3)}</title></circle>)}<text x="314" y="244" textAnchor="middle" fill="var(--foreground)" fontSize="12">Generation step</text><text x="16" y="112" transform="rotate(-90 16 112)" textAnchor="middle" fill="var(--foreground)" fontSize="12">Mode entropy</text><text x="50" y="30" textAnchor="end" fill="var(--muted-foreground)" fontSize="11">{trajectoryMax.toFixed(2)}</text><text x="50" y="208" textAnchor="end" fill="var(--muted-foreground)" fontSize="11">0</text></svg><figcaption className="px-2 pb-1 text-xs text-muted-foreground">Persistent entropy after each active-generation update; the line begins when this mode first enters the bank.</figcaption></figure>
        <div className="space-y-4"><div className="grid grid-cols-2 gap-3">{[["Final entropy",selected.finalEntropy?.toFixed(3) ?? "—"],["Entropy change",`${selected.entropyChange >= 0 ? "+" : ""}${selected.entropyChange.toFixed(3)}`],["Generated",formatNumber(selected.generated)],[`${activeModel || "Model"} selected`,selectionAvailable ? formatNumber(selected.selected) : "—"]].map(([label,value]) => <div key={label} className="rounded-xl border bg-background p-3"><p className="text-[11px] text-muted-foreground">{label}</p><p className="mt-1 font-mono text-lg font-semibold">{value}</p></div>)}</div><div><div className="mb-1.5 flex justify-between text-xs"><span>Generation contribution</span><span className="font-mono">{totalGenerated ? (selected.generated / totalGenerated * 100).toFixed(1) : "0.0"}%</span></div><div className="h-2 rounded-full bg-muted"><div className="h-full rounded-full bg-[var(--split-train)]" style={{width:`${totalGenerated ? selected.generated / totalGenerated * 100 : 0}%`}} /></div></div>{selectionAvailable && <div><div className="mb-1.5 flex justify-between text-xs"><span>Selection weight contribution</span><span className="font-mono">{totalWeight ? (selected.totalWeight / totalWeight * 100).toFixed(1) : "0.0"}%</span></div><div className="h-2 rounded-full bg-muted"><div className="h-full rounded-full bg-[var(--split-valid)]" style={{width:`${totalWeight ? selected.totalWeight / totalWeight * 100 : 0}%`}} /></div><p className="mt-1.5 text-[11px] text-muted-foreground">{(selected.fractionSelected * 100).toFixed(1)}% of agents in this mode selected · average fitted weight {selected.averageWeight.toFixed(4)}</p></div>}{selected.attributes.length > 0 && <div><p className="mb-2 text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">Contributing attributes</p><div className="flex max-h-28 flex-wrap gap-1.5 overflow-auto">{selected.attributes.map((attribute) => <span key={attribute} className="rounded-md bg-muted px-2 py-1 text-[11px]">{attribute}</span>)}</div></div>}</div>
      </div>
    </section>}
  </div>;
}

function ReportResult({ report, detail, annotation, onAnnotation, onBack }: { report: Report; detail: ReportDetail; annotation?: Annotation; onAnnotation: (annotation: Annotation) => void; onBack: () => void }) {
  const [tab, setTab] = useState<"overview"|"predictions"|"diagnostics"|"generation"|"reports"|"files">("overview");
  const [documentName, setDocumentName] = useState(detail.documents[0]?.name || "");
  const document = detail.documents.find((item) => item.name === documentName) || detail.documents[0];
  const images = (category: ReportDetail["images"][number]["category"]) => detail.images.filter((image) => image.category === category);
  const highlights = [["Endowments", report.numEndowments], ["Effective rank", Number(detail.results.effective_rank)], ["Final entropy", Number(detail.results.final_entropy)], ["Total cost", detail.tokenUsage.total_cost ?? report.totalCost]] as const;
  return <>
    <Button variant="ghost" className="mb-4 -ml-3 text-muted-foreground" onClick={onBack}><ArrowLeft /> Results library</Button>
    <Header eyebrow={`${report.kind} report / ${report.path}`} title={report.name} copy={`${report.title} · ${report.reportCount} Markdown reports · ${report.imageCount} figures · updated ${formatDate(report.modifiedAt)}`} />
    <AnnotationEditor path={report.path} annotation={annotation} onSaved={onAnnotation} />
    <Tabs value={tab} onChange={setTab} items={[{id:"overview",label:"Overview"},{id:"predictions",label:"Predictions",count:images("predictions").length},{id:"diagnostics",label:"Diagnostics",count:images("diagnostics").length},{id:"generation",label:"Generation",count:images("generation").length},{id:"reports",label:"Reports",count:detail.documents.length},{id:"files",label:"Files",count:detail.files.length}]} />
    {tab === "overview" ? (
      <div className="space-y-5"><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{highlights.map(([label,value]) => <div key={label} className="surface-card p-5"><p className="text-xs font-semibold uppercase tracking-[.08em] text-muted-foreground">{label}</p><p className="mt-2 text-2xl font-semibold">{label === "Total cost" ? formatMoney(Number(value)) : formatNumber(Number(value),label === "Endowments" ? 0 : 3)}</p></div>)}</div><div className="grid gap-5 xl:grid-cols-[1fr_1.2fr]"><section className="surface-card p-5 sm:p-6"><div className="mb-5"><h2 className="section-title">Model comparison</h2><p className="section-copy">Test-set metrics recovered from summary_aggregate.csv.</p></div><ReportMetrics report={report} /></section>{document ? <section className="surface-card max-h-[720px] overflow-auto p-5 sm:p-6"><MarkdownDocument content={document.content} /></section> : <div />}</div></div>
    ) : tab === "predictions" ? (
      <ImageGallery images={images("predictions")} empty="No prediction figures were written for this experiment." />
    ) : tab === "diagnostics" ? (
      <ImageGallery images={images("diagnostics")} empty="No model diagnostic figures were written for this experiment." />
    ) : tab === "generation" ? (
      <div className="space-y-5"><ModeEntropyAnalysis entropyRows={detail.modeEntropyRows} summaryRows={detail.modeSummaryRows} selectionRows={detail.modeRows} selectionSource={detail.modeSelectionSource} preferredModel={report.models[0]} /><EntropyTrajectory rows={detail.aggregateEntropy} /><ImageGallery images={images("generation")} empty="No generation or entropy figures were written for this experiment." /></div>
    ) : tab === "reports" ? (
      <div className="grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)]"><aside className="surface-card self-start overflow-hidden">{detail.documents.map((item) => <button key={item.name} onClick={() => setDocumentName(item.name)} className={`flex w-full items-center gap-3 border-b px-4 py-3 text-left text-sm last:border-0 ${document?.name === item.name ? "bg-primary/8 text-primary" : "hover:bg-muted"}`}><FileText className="size-4 shrink-0" /><span className="min-w-0"><strong className="block truncate">{item.title}</strong><span className="block truncate font-mono text-[11px] text-muted-foreground">{item.name}</span></span></button>)}</aside><article className="surface-card min-w-0 p-5 sm:p-7">{document && <MarkdownDocument content={document.content} />}</article></div>
    ) : (
      <section className="surface-card overflow-hidden"><div className="border-b px-5 py-4"><h2 className="section-title">Artifact inventory</h2><p className="section-copy">Every downloadable file written into this experiment directory.</p></div>{detail.files.map((file) => <div key={file.name} className="grid grid-cols-[minmax(0,1fr)_90px_90px_44px] items-center gap-3 border-b px-5 py-3 text-sm last:border-0"><span className="truncate font-mono text-xs">{file.name}</span><span className="text-muted-foreground">{file.type}</span><span className="text-right font-mono text-xs">{formatBytes(file.size)}</span><a href={fileUrl(`${report.path}/${file.name}`)} download aria-label={`Download ${file.name}`} className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground"><Download className="size-4" /></a></div>)}</section>
    )}
  </>;
}

export function ResultsView({ data, liveRuns, focusRunPath, focusReportPath, focusComparePaths, onFocusHandled, onRouteChange }: { data: ResultsData; liveRuns?: ReactNode; focusRunPath?: string | null; focusReportPath?: string | null; focusComparePaths?: string[]; onFocusHandled?: () => void; onRouteChange?: (selection: { run?: string; report?: string }) => void }) {
  const catalog = metricCatalog(data);
  const [source, setSource] = useState<"reports"|"canonical"|"aggregates">("canonical");
  const [visibleAggregates, setVisibleAggregates] = useState<AggregateSummary[]>([]);
  const [selectedAggregate, setSelectedAggregate] = useState<AggregateSummary | null>(null);
  const [aggregateDetail, setAggregateDetail] = useState<AggregateDetail | null>(null);
  const [query, setQuery] = useState("");
  const [lineage, setLineage] = useState<"all"|"current"|"legacy">("all");
  const [sort, setSort] = useState<"newest"|"pinned"|"mse"|"one_minus_jsd"|"emd"|"mcpa">("newest");
  const [reportSort, setReportSort] = useState<"newest"|"pinned">("newest");
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [listLoading, setListLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [visibleRuns, setVisibleRuns] = useState<Run[]>(data.runs.slice(0, 30));
  const [visibleReports, setVisibleReports] = useState<Report[]>(data.reports.slice(0, 30));
  const [recordCount, setRecordCount] = useState(data.workspace.counts.runs);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reportDetail, setReportDetail] = useState<ReportDetail | null>(null);
  const [comparison, setComparison] = useState<Run[]>([]);
  const [annotations, setAnnotations] = useState<Record<string, Annotation>>(data.annotations || {});
  const currentRunCount = data.workspace.counts.currentRuns;
  const historicalRunCount = data.workspace.counts.historicalRuns;
  const pageSize = 30;
  const pageCount = Math.max(1, Math.ceil(recordCount / pageSize));
  const runGroups = useMemo(() => {
    const groups = new Map<string, { key: string; label: string; runs: Run[] }>();
    visibleRuns.forEach((run) => {
      const family = runFamily(run);
      const group = groups.get(family.key);
      if (group) group.runs.push(run);
      else groups.set(family.key, { key: family.key, label: family.label, runs: [run] });
    });
    return [...groups.values()];
  }, [visibleRuns]);
  useEffect(() => { setPage(0); }, [lineage, query, sort, reportSort, source]);
  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(() => {
      setListLoading(true);
      const endpoint = source === "canonical"
        ? `/api/runs?cursor=${page * pageSize}&limit=${pageSize}&origin=${lineage}&sort=${sort}&query=${encodeURIComponent(query)}`
        : source === "aggregates"
          ? `/api/aggregates?cursor=${page * pageSize}&limit=${pageSize}&query=${encodeURIComponent(query)}`
          : `/api/reports?cursor=${page * pageSize}&limit=${pageSize}&sort=${reportSort}&query=${encodeURIComponent(query)}`;
      void getJson<{ items: Run[] | Report[] | AggregateSummary[]; total: number; indexing?: boolean }>(endpoint).then((payload) => {
        if (!active) return;
        if (source === "canonical") setIndexing(Boolean(payload.indexing));
        if (source === "canonical") setVisibleRuns(payload.items as Run[]);
        else if (source === "aggregates") setVisibleAggregates(payload.items as AggregateSummary[]);
        else setVisibleReports(payload.items as Report[]);
        setRecordCount(payload.total);
      }).catch((error) => {
        if (active) toast.error(error instanceof Error ? error.message : "Could not load the research library.");
      }).finally(() => { if (active) setListLoading(false); });
    }, query ? 220 : 0);
    return () => { active = false; window.clearTimeout(timer); };
  }, [source, page, lineage, query, sort, reportSort]);
  async function openRun(run: Run) { setSelectedRun(run); setLoading(true); try { setRunDetail(await getJson<RunDetail>(`/api/run?path=${encodeURIComponent(run.path)}`)); onRouteChange?.({ run: run.path }); } catch(error) { toast.error(error instanceof Error ? error.message : "Could not load run."); setSelectedRun(null); } finally { setLoading(false); } }
  async function openAggregate(item: AggregateSummary) { setSelectedAggregate(item); setLoading(true); try { setAggregateDetail(await getJson<AggregateDetail>(`/api/aggregate?path=${encodeURIComponent(item.path)}`)); } catch(error) { toast.error(error instanceof Error ? error.message : "Could not load this summary."); setSelectedAggregate(null); } finally { setLoading(false); } }
  useEffect(() => {
    if (!focusComparePaths?.length) return;
    let active = true;
    void Promise.all(focusComparePaths.map((path) => getJson<RunDetail>(`/api/run?path=${encodeURIComponent(path)}`)))
      .then((details) => {
        if (!active) return;
        const runs = details.map((detail) => detail.summary).filter((run): run is Run => Boolean(run));
        setSource("canonical");
        setComparison(runs.slice(0, 4));
        onFocusHandled?.();
      })
      .catch((error) => { if (active) { onFocusHandled?.(); toast.error(error instanceof Error ? error.message : "Could not open the comparison."); } });
    return () => { active = false; };
  }, [focusComparePaths?.join("|")]);
  async function openReport(report: Report) { setSelectedReport(report); setLoading(true); try { setReportDetail(await getJson<ReportDetail>(`/api/report?path=${encodeURIComponent(report.path)}`)); onRouteChange?.({ report: report.path }); } catch(error) { toast.error(error instanceof Error ? error.message : "Could not load report."); setSelectedReport(null); } finally { setLoading(false); } }
  useEffect(() => {
    if (!focusRunPath) return;
    let active = true;
    void (async () => {
      let run = data.runs.find((item) => item.path === focusRunPath);
      if (!run) {
        const payload = await getJson<{ items: Run[] }>(`/api/runs?limit=20&query=${encodeURIComponent(focusRunPath.split("/").at(-1) || focusRunPath)}`);
        run = payload.items.find((item) => item.path === focusRunPath);
      }
      if (!run || !active) throw new Error("The completed demo result was not found in the refreshed library.");
      setSelectedRun(run);
      setLoading(true);
      try {
        const detail = await getJson<RunDetail>(`/api/run?path=${encodeURIComponent(run.path)}`);
        if (active) {
          setRunDetail(detail);
          onFocusHandled?.();
        }
      } finally {
        if (active) setLoading(false);
      }
    })().catch((error) => {
      if (active) {
        setSelectedRun(null);
        setLoading(false);
        onFocusHandled?.();
        toast.error(error instanceof Error ? error.message : "Could not open the offline demo result.");
      }
    });
    return () => { active = false; };
  }, [focusRunPath]);
  useEffect(() => {
    if (!focusReportPath) return;
    let active = true;
    void (async () => {
      let report = data.reports.find((item) => item.path === focusReportPath);
      if (!report) {
        const payload = await getJson<{ items: Report[] }>(`/api/reports?limit=20&query=${encodeURIComponent(focusReportPath.split("/").at(-1) || focusReportPath)}`);
        report = payload.items.find((item) => item.path === focusReportPath);
      }
      if (!report || !active) throw new Error("The requested report was not found in the library.");
      setSelectedReport(report);
      setLoading(true);
      try {
        const detail = await getJson<ReportDetail>(`/api/report?path=${encodeURIComponent(report.path)}`);
        if (active) { setReportDetail(detail); onFocusHandled?.(); }
      } finally { if (active) setLoading(false); }
    })().catch((error) => {
      if (active) { setSelectedReport(null); setLoading(false); onFocusHandled?.(); toast.error(error instanceof Error ? error.message : "Could not open report."); }
    });
    return () => { active = false; };
  }, [focusReportPath]);
  const back = () => { setSelectedRun(null); setRunDetail(null); setSelectedReport(null); setReportDetail(null); setSelectedAggregate(null); setAggregateDetail(null); onRouteChange?.({}); };
  const toggleComparison = (run: Run) => setComparison((current) => {
    if (current.some((item) => item.path === run.path)) return current.filter((item) => item.path !== run.path);
    if (current.length >= 4) { toast.error("Compare up to four runs at a time."); return current; }
    return [...current, run];
  });
  const saveAnnotation = (path: string, annotation: Annotation) => {
    setAnnotations((current) => ({ ...current, [path]: annotation }));
    setVisibleRuns((current) => current.map((run) => run.path === path ? { ...run, annotation } : run));
    setVisibleReports((current) => current.map((report) => report.path === path ? { ...report, annotation } : report));
    setSelectedRun((current) => current?.path === path ? { ...current, annotation } : current);
    setSelectedReport((current) => current?.path === path ? { ...current, annotation } : current);
  };
  if (loading) return <Loading label="Consolidating experiment artifacts…" />;
  if (selectedAggregate && aggregateDetail) return <AggregateResult detail={aggregateDetail} catalog={catalog} onBack={back} />;
  if (selectedRun && runDetail) return <CanonicalRun run={selectedRun} detail={runDetail} catalog={catalog} annotation={annotations[selectedRun.path] || selectedRun.annotation} onAnnotation={(annotation) => saveAnnotation(selectedRun.path, annotation)} onBack={back} />;
  if (selectedReport && reportDetail) return <ReportResult report={selectedReport} detail={reportDetail} annotation={annotations[selectedReport.path] || selectedReport.annotation} onAnnotation={(annotation) => saveAnnotation(selectedReport.path, annotation)} onBack={back} />;
  const pager = recordCount > pageSize && <div className="flex items-center justify-between border-t bg-muted/25 px-5 py-3 text-sm"><span className="text-muted-foreground">Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, recordCount)} of {recordCount}</span><div className="flex items-center gap-2"><Button variant="outline" size="sm" disabled={page === 0 || listLoading} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</Button><span className="min-w-16 text-center font-mono text-xs text-muted-foreground">{listLoading ? "Loading" : `${page + 1} / ${pageCount}`}</span><Button variant="outline" size="sm" disabled={page + 1 >= pageCount || listLoading} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>Next</Button></div></div>;

  return <>
    <Header eyebrow="Workspace / Runs" title="Every run, in one place" copy={historicalRunCount || data.workspace.counts.reports ? `${currentRunCount} runs in the current format, ${historicalRunCount} in the earlier format, and ${data.workspace.counts.reports} result folders.` : `${currentRunCount} ${currentRunCount === 1 ? "run" : "runs"} recorded under outputs/.`} />
    {liveRuns}
    <div className="mb-5 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
      <Tabs value={source} onChange={setSource} items={[{id:"canonical",label:"P2P runs",count:data.workspace.counts.runs},{id:"aggregates",label:"Pooled summaries",count:data.workspace.counts.aggregates ?? 0},...(data.workspace.counts.reports ? [{id:"reports" as const,label:"Result folders",count:data.workspace.counts.reports}] : [])]} />
      <div className="flex w-full flex-col gap-2 sm:flex-row xl:w-auto">
        {source === "canonical" && <>{historicalRunCount > 0 && <select aria-label="Filter run lineage" value={lineage} onChange={(event) => setLineage(event.target.value as typeof lineage)} className="h-10 rounded-md border bg-card px-3 text-sm"><option value="all">All formats</option><option value="current">Current · {currentRunCount}</option><option value="legacy">Earlier format · {historicalRunCount}</option></select>}<select aria-label="Sort runs" value={sort} onChange={(event) => setSort(event.target.value as typeof sort)} className="h-10 rounded-md border bg-card px-3 text-sm"><option value="newest">Newest first</option><option value="pinned">Pinned first</option><option value="mse">Best MSE</option><option value="one_minus_jsd">Best 1 − JSD</option><option value="emd">Best EMD</option><option value="mcpa">Best MCPA</option></select></>}
        {source === "reports" && <select aria-label="Sort reports" value={reportSort} onChange={(event) => setReportSort(event.target.value as typeof reportSort)} className="h-10 rounded-md border bg-card px-3 text-sm"><option value="newest">Newest first</option><option value="pinned">Pinned first</option></select>}
        <div className="relative w-full sm:w-80"><Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input aria-label="Search results" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name, path, model, or lineage…" className="bg-card pl-9" /></div>
      </div>
    </div>
    {source === "canonical" && indexing && <div className="mb-5 flex items-center gap-2 rounded-xl border bg-muted/30 px-4 py-3 text-sm text-muted-foreground"><LoaderCircle className="size-4 animate-spin" />Studio is building its run index in the background. Newest-first browsing works now; sorting by a metric and search wait for the index, which is built once and then kept up to date.</div>}
    {source === "canonical" && comparison.length > 0 && comparison.length < 2 && <div className="mb-5 flex items-center justify-between rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm"><span><strong>{comparison[0].runId}</strong> selected. Choose one more run to compare.</span><Button variant="ghost" size="sm" onClick={() => setComparison([])}>Clear</Button></div>}
    {source === "canonical" && comparison.length >= 2 && <RunComparison runs={comparison} catalog={catalog} onRemove={(path) => setComparison((current) => current.filter((item) => item.path !== path))} onClear={() => setComparison([])} />}
    {source === "canonical" && <PopulationFamilyLegend />}
    {source === "aggregates" ? (
      !visibleAggregates.length && !listLoading ? <div className="surface-card p-10 text-center text-sm text-muted-foreground">No pooled summaries yet. Run <code className="rounded bg-muted px-1">p2p aggregate</code> on a set of runs and its output appears here.</div> : <section className={`surface-card overflow-hidden ${listLoading ? "opacity-60" : ""}`}><div className="hidden grid-cols-[minmax(260px,1.6fr)_90px_80px_80px_minmax(160px,1fr)_100px_24px] gap-3 border-b bg-muted/40 px-5 py-3 text-xs font-semibold uppercase tracking-[.06em] text-muted-foreground lg:grid"><span>Summary</span><span>Status</span><span>Runs</span><span>Groups</span><span>Grouped by</span><span>Updated</span><span /></div>{visibleAggregates.map((item) => <AggregateRow key={item.path} item={item} onOpen={() => void openAggregate(item)} />)}{pager}</section>
    ) : source === "reports" ? (
      !visibleReports.length && !listLoading ? <div className="surface-card p-10 text-center text-sm text-muted-foreground">No reports match this search.</div> : <section className={`surface-card overflow-hidden ${listLoading ? "opacity-60" : ""}`}><div className="hidden grid-cols-[minmax(260px,1.6fr)_100px_130px_90px_90px_100px_24px] gap-3 border-b bg-muted/40 px-5 py-3 text-xs font-semibold uppercase tracking-[.06em] text-muted-foreground lg:grid"><span>Experiment</span><span>Type</span><span>Models</span><span>Test MSE</span><span>Figures</span><span>Updated</span><span /></div>{visibleReports.map((report) => { const annotation = annotations[report.path] || report.annotation; return <button key={report.path} onClick={() => void openReport(report)} className="grid w-full gap-2 border-b px-5 py-4 text-left transition-colors last:border-0 hover:bg-muted/45 lg:grid-cols-[minmax(260px,1.6fr)_100px_130px_90px_90px_100px_24px] lg:items-center lg:gap-3"><span className="min-w-0"><span className="flex min-w-0 items-center gap-2"><strong className="truncate text-sm">{report.name}</strong>{annotation?.pinned && <Star className="size-3.5 shrink-0 fill-current text-amber-500" />}{annotation?.label && <Badge variant="secondary" className="max-w-40 truncate text-[10px]">{annotation.label}</Badge>}</span><span className="block truncate font-mono text-[11px] text-muted-foreground">{report.path}</span></span><Badge variant="outline" className="w-fit capitalize">{report.kind}</Badge><span className="truncate text-sm">{report.models.join(", ") || "—"}</span><span className="font-mono text-sm">{formatNumber(report.metrics[0]?.testMse,4)}</span><span className="text-sm"><FileImage className="mr-1 inline size-3.5 text-muted-foreground" />{report.imageCount}</span><span className="text-xs text-muted-foreground">{formatDate(report.modifiedAt)}</span><ChevronRight className="hidden size-4 text-muted-foreground lg:block" /></button>; })}{pager}</section>
    ) : !visibleRuns.length && !listLoading ? (
      <div className="surface-card p-10 text-center text-sm text-muted-foreground">No P2P runs match these filters.</div>
    ) : (
      <section className={`surface-card overflow-hidden ${listLoading ? "opacity-60" : ""}`}>
        <div className="hidden grid-cols-[32px_minmax(260px,1.6fr)_120px_repeat(4,90px)_100px_24px] gap-3 border-b bg-muted/40 px-5 py-3 text-xs font-semibold uppercase tracking-[.06em] text-muted-foreground lg:grid"><span aria-label="Compare" /><span>Run</span><span>Model</span><span>MSE</span><span>1 − JSD</span><span>EMD</span><span>MCPA</span><span>Updated</span><span /></div>
        {runGroups.map((group) => <section key={group.key} className="run-family-group"><header className="run-family-divider"><span>{group.label}</span><small>{group.runs.length} {group.runs.length === 1 ? "record" : "repetitions"}</small></header><div className="run-family-runs">{group.runs.map((run) => <RunLibraryRow key={run.path} run={run} annotation={annotations[run.path] || run.annotation} selected={comparison.some((item) => item.path === run.path)} onCompare={() => toggleComparison(run)} onOpen={() => void openRun(run)} />)}</div></section>)}
        {pager}
      </section>
    )}
  </>;
}
