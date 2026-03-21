/**
 * API client for connecting frontend to the FastAPI backend.
 * All functions point at the backend's /api/v1 endpoints.
 */

const API_BASE = "http://localhost:8000/api/v1";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }

  return res.json();
}

// ── Dashboard ────────────────────────────────────────────────────────────────

export interface DashboardStats {
  kpis: {
    avgScore: number;
    totalCalls: number;
    resolutionRate: number;
    violationCount: number;
  };
  weeklyTrend: Array<{ day: string; score: number }>;
  emotionDistribution: Array<{ name: string; value: number; color: string }>;
  policyCompliance: Array<{ category: string; rate: number; color: string }>;
  agentPerformance: Array<{
    name: string;
    empathy: number;
    policy: number;
    resolution: number;
    overallScore: number;
    trend: "up" | "down";
  }>;
  interactions: InteractionSummary[];
}

export function getDashboardStats(): Promise<DashboardStats> {
  return apiFetch<DashboardStats>("/dashboard/stats");
}

// ── Interactions ──────────────────────────────────────────────────────────────

export interface InteractionSummary {
  id: string;
  agentName: string;
  agentId: string;
  date: string;
  time: string;
  duration: string;
  language: string;
  overallScore: number;
  empathyScore: number;
  policyScore: number;
  resolutionScore: number;
  resolved: boolean;
  hasViolation: boolean;
  hasOverlap: boolean;
  responseTime: string;
  status: string;
  audioFilePath?: string | null;
}

export function getInteractions(): Promise<InteractionSummary[]> {
  return apiFetch<InteractionSummary[]>("/interactions");
}

export interface UtteranceData {
  id: string;
  interactionId: string;
  speaker: string;
  text: string;
  startTime: number;
  endTime: number;
  timestamp: string;
  emotion: string;
  confidence: number;
}

export interface EmotionEventData {
  id: string;
  interactionId: string;
  previousEmotion: string;
  newEmotion: string;
  fromEmotion: string;
  toEmotion: string;
  jumpToSeconds: number;
  timestamp: string;
  confidenceScore: number;
  delta: number;
  speaker: string;
  llmJustification: string;
  justification: string;
}

export interface PolicyViolationData {
  id: string;
  interactionId: string;
  policyName: string;
  policyTitle: string;
  category: string;
  description: string;
  reasoning: string;
  severity: string;
  score: number;
  timestamp?: string;
}

export interface InteractionDetailInfo extends InteractionSummary {
  audioFilePath?: string | null;
}

export interface EmotionDistribution {
  emotion: string;
  count: number;
  pct: number;
}

export interface EmotionComparison {
  interactionId?: string;
  totalUtterances: number;
  distributions: {
    acoustic: EmotionDistribution[];
    text: EmotionDistribution[];
    fused: EmotionDistribution[];
  };
  quality: {
    acousticTextAgreementRate: number;
    fusedMatchesAcousticRate: number;
    fusedMatchesTextRate: number;
    disagreementCount: number;
  };
  evidence?: {
    emotionShiftQuotes?: string[];
    processAdherenceQuotes?: string[];
    nliPolicyQuotes?: string[];
    citations?: Array<{
      source: "acoustic" | "text" | "fused";
      speaker?: string;
      quote: string;
      utteranceIndex?: number;
    }>;
  };
}

export interface LLMEvidenceCitation {
  source: "transcript" | "policy" | "sop" | "acoustic";
  speaker?: "customer" | "agent" | "system" | "unknown";
  quote: string;
  utteranceIndex?: number | null;
}

export interface LLMEmotionShift {
  isDissonanceDetected: boolean;
  dissonanceType: string;
  rootCause: string;
  counterfactualCorrection: string;
  evidenceQuotes: string[];
  citations: LLMEvidenceCitation[];
}

export interface LLMProcessAdherence {
  detectedTopic: string;
  isResolved: boolean;
  efficiencyScore: number;
  justification: string;
  missingSopSteps: string[];
  evidenceQuotes: string[];
  citations: LLMEvidenceCitation[];
}

export interface LLMNliPolicy {
  nliCategory: "Entailment" | "Benign Deviation" | "Contradiction" | "Policy Hallucination";
  justification: string;
  evidenceQuotes: string[];
  citations: LLMEvidenceCitation[];
}

export interface LLMTriggerReport {
  available: boolean;
  error?: string;
  orgFilter?: string | null;
  forcedRerun?: boolean;
  interactionId?: string;
  emotionShift?: LLMEmotionShift;
  processAdherence?: LLMProcessAdherence;
  nliPolicy?: LLMNliPolicy;
  derived?: {
    customerText: string;
    acousticEmotion: string;
    fusedEmotion: string;
    agentStatement: string;
  };
}

export interface InteractionDetail {
  interaction: InteractionDetailInfo;
  utterances: UtteranceData[];
  emotionEvents: EmotionEventData[];
  policyViolations: PolicyViolationData[];
  emotionComparison?: EmotionComparison;
  llmTriggers?: LLMTriggerReport | null;
}

export function getInteractionDetail(
  id: string,
  options?: { includeLLMTriggers?: boolean; llmOrgFilter?: string; llmForceRerun?: boolean },
): Promise<InteractionDetail> {
  const params = new URLSearchParams();
  if (options?.includeLLMTriggers) {
    params.set("include_llm_triggers", "true");
  }
  if (options?.llmOrgFilter) {
    params.set("llm_org_filter", options.llmOrgFilter);
  }
  if (options?.llmForceRerun) {
    params.set("llm_force_rerun", "true");
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return apiFetch<InteractionDetail>(`/interactions/${id}${suffix}`);
}

export function getAudioUrl(interactionId: string): string {
  return `${API_BASE}/interactions/${interactionId}/audio`;
}

// ── Knowledge Base ───────────────────────────────────────────────────────────

export interface PolicyData {
  id: string;
  title: string;
  category: string;
  content: string;
  preview: string;
  lastUpdated: string;
  isActive: boolean;
}

export function getPolicies(): Promise<PolicyData[]> {
  return apiFetch<PolicyData[]>("/knowledge/policies");
}

export interface FAQData {
  id: string;
  question: string;
  answer: string;
  preview: string;
  category: string;
  isActive: boolean;
}

export function getFaqs(): Promise<FAQData[]> {
  return apiFetch<FAQData[]>("/knowledge/faqs");
}

// ── Agents ───────────────────────────────────────────────────────────────────

export interface AgentSummary {
  id: string;
  name: string;
  role: string;
}

export function getAgents(): Promise<AgentSummary[]> {
  return apiFetch<AgentSummary[]>("/agents");
}

export interface AgentProfile {
  id: string;
  name: string;
  role: string;
  totalCalls: number;
  callsThisWeek: number;
  teamRank: number;
  avgScore: number;
  overallScore: number;
  empathyScore: number;
  policyScore: number;
  resolutionScore: number;
  resolutionRate: number;
  avgResponseTime: string;
  trend: "up" | "down";
  weeklyTrend: Array<{ day: string; score: number }>;
  recentCalls: Array<{
    id: string;
    date: string;
    time: string;
    score: number;
    duration: string;
    language: string;
    resolved: boolean;
    hasReview: boolean;
  }>;
}

export function getAgentProfile(agentId: string): Promise<AgentProfile> {
  return apiFetch<AgentProfile>(`/agents/${agentId}`);
}

// ── Assistant ────────────────────────────────────────────────────────────────

export interface AssistantResponse {
  id: string;
  type: "user" | "ai";
  content: string;
  mode: string;
  sql?: string;
  execution_time?: string;
  data?: any[];
  success: boolean;
}

export function sendAssistantQuery(text: string, mode: "chat" | "voice" = "chat"): Promise<AssistantResponse> {
  return apiFetch<AssistantResponse>(`/assistant/query`, {
    method: "POST",
    body: JSON.stringify({
      query_text: text,
      mode: mode,
    }),
  });
}

// ── RAG Service ──────────────────────────────────────────────────────────────

export interface RAGQueryRequest {
  query: string;
  mode?: "answer" | "compliance";
  org_filter?: string | null;
}

export interface RAGQueryResponse {
  response: string;
  chunks: Array<any>;
  timing: any;
}

export function queryRag(data: RAGQueryRequest): Promise<RAGQueryResponse> {
  return apiFetch<RAGQueryResponse>("/rag/query", {
    method: "POST",
    body: JSON.stringify(data),
  });
}
