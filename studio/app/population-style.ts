export const populationFamilyColors = {
  Core: "#236b8e",
  Thematic: "#4f8a43",
  Theoretical: "#d89c00",
  Survey: "#7b61a8",
  "Question patch": "#e7811d",
  "Mixed patch": "#c43d3d",
} as const;

export type PopulationFamily = keyof typeof populationFamilyColors;

/**
 * Prefer the family recorded by the Studio API. The fallback keeps historical
 * records useful when no attribute-bank file survives beside the run.
 */
export function populationFamily(mode: string, recordedFamily?: string): PopulationFamily {
  if (recordedFamily && recordedFamily in populationFamilyColors) {
    return recordedFamily as PopulationFamily;
  }

  const tokens = mode.split("+").map((token) => token.trim()).filter(Boolean);
  const lowered = tokens.map((token) => token.toLowerCase());
  const hasSurvey = lowered.includes("survey");
  const isQuestionPatch = (token: string) => /(?:^|_)(?:gap\d*q|p?q|rq|scm|f\d)/i.test(token) || /_w\d+$/i.test(token);
  const hasQuestion = tokens.some(isQuestionPatch);
  const theoretical = new Set(["maslow", "big_five"]);
  const thematic = new Set(["economics", "politics", "values_beliefs", "cognition_psychology", "information", "culture_history", "behavioral_biases"]);

  if (tokens.length > 1 && hasQuestion) return "Mixed patch";
  if (tokens.length === 1 && hasQuestion) return "Question patch";
  if (lowered.length === 1 && lowered[0] === "core") return "Core";
  if (hasSurvey) return "Survey";
  if (lowered.some((token) => theoretical.has(token))) return "Theoretical";
  if (lowered.some((token) => thematic.has(token))) return "Thematic";
  return "Thematic";
}
