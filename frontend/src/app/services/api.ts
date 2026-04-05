/**
 * API client for connecting frontend to the FastAPI backend.
 * All functions point at the backend's /api/v1 endpoints.
 */

const API_ROOT = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_BASE = `${API_ROOT.replace(/\/$/, "")}/api/v1`;

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const isFormData = typeof FormData !== "undefined" && options?.body instanceof FormData;
  const headers = new Headers(options?.headers || {});
  if (!isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    ...options,
    headers,
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

export interface CreateInteractionResult {
  interactionId: string;
  status: string;
  audioFilePath: string;
  agentId: string;
  uploadedBy: string;
  processingJobs: Array<{
    stage: string;
    status: string;
    retryCount: number;
    errorMessage?: string | null;
  }>;
}

export function createInteraction(file: File, agentId?: string): Promise<CreateInteractionResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (agentId) {
    formData.append("agent_id", agentId);
  }
  return apiFetch<CreateInteractionResult>("/interactions", {
    method: "POST",
    body: formData,
  });
}

export interface ProcessingStatusResult {
  interactionId: string;
  status: string;
  jobs: Array<{
    stage: string;
    status: string;
    retryCount: number;
    errorMessage?: string | null;
    startedAt?: string | null;
    completedAt?: string | null;
  }>;
}

export function getInteractionProcessingStatus(id: string): Promise<ProcessingStatusResult> {
  return apiFetch<ProcessingStatusResult>(`/interactions/${id}/processing-status`);
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
  textEmotion?: string;
  textConfidence?: number;
  fusedEmotion?: string;
  fusedConfidence?: number;
  fusionModel?: string;
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
  currentCustomerEmotion?: string;
  currentEmotionReasoning?: string;
  counterfactualCorrection: string;
  evidenceQuotes: string[];
  citations: LLMEvidenceCitation[];
  insufficientEvidence?: boolean;
}

export interface LLMProcessAdherence {
  detectedTopic: string;
  isResolved: boolean;
  efficiencyScore: number;
  justification: string;
  missingSopSteps: string[];
  evidenceQuotes: string[];
  citations: LLMEvidenceCitation[];
  insufficientEvidence?: boolean;
}

export interface LLMNliPolicy {
  nliCategory: "Entailment" | "Benign Deviation" | "Contradiction" | "Policy Hallucination";
  justification: string;
  evidenceQuotes: string[];
  citations: LLMEvidenceCitation[];
  policyVersion?: string | null;
  policyEffectiveAt?: string | null;
  policyCategory?: string | null;
  conflictResolutionApplied?: boolean;
  insufficientEvidence?: boolean;
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

export interface EmotionTriggerReport {
  available: boolean;
  error?: string;
  orgFilter?: string | null;
  forcedRerun?: boolean;
  interactionId?: string;
  emotionShift?: LLMEmotionShift;
  derived?: {
    customerText: string;
    acousticEmotion: string;
    fusedEmotion: string;
    agentStatement: string;
  };
}

export interface RagComplianceReport {
  available: boolean;
  error?: string;
  orgFilter?: string | null;
  forcedRerun?: boolean;
  interactionId?: string;
  processAdherence?: LLMProcessAdherence;
  nliPolicy?: LLMNliPolicy;
  policyViolations?: PolicyViolationData[];
}

export interface InteractionDetail {
  interaction: InteractionDetailInfo;
  utterances: UtteranceData[];
  emotionEvents: EmotionEventData[];
  policyViolations: PolicyViolationData[];
  emotionComparison?: EmotionComparison;
  ragCompliance?: RagComplianceReport | null;
  emotionTriggers?: EmotionTriggerReport | null;
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

export interface ReprocessResult {
  interactionId: string;
  status: string;
  queued: boolean;
  processingJobs: Array<{
    stage: string;
    status: string;
    retryCount: number;
    errorMessage?: string | null;
  }>;
}

export function reprocessInteraction(id: string, options?: { force?: boolean }): Promise<ReprocessResult> {
  const suffix = options?.force ? "?force=true" : "";
  return apiFetch<ReprocessResult>(`/interactions/${id}/reprocess${suffix}`, {
    method: "POST",
  });
}

// ── Knowledge Base ───────────────────────────────────────────────────────────

export interface PolicyData {
  id: string;
  documentType: "policy";
  title: string;
  category: string;
  content: string;
  preview: string;
  lastUpdated: string;
  isActive: boolean;
  usageCount: number;
}

export function getPolicies(): Promise<PolicyData[]> {
  return apiFetch<PolicyData[]>("/knowledge/policies");
}

export function uploadPolicyDocument(data: { title?: string; category?: string; file: File }): Promise<{ id: string }> {
  const formData = new FormData();
  if (data.title) formData.append("title", data.title);
  if (data.category) formData.append("category", data.category);
  formData.append("file", data.file);

  return apiFetch<{ id: string }>("/knowledge/policies/upload", {
    method: "POST",
    body: formData,
  });
}

export function replacePolicyDocument(id: string, data: { title?: string; category?: string; file: File }): Promise<{ id: string }> {
  const formData = new FormData();
  if (data.title) formData.append("title", data.title);
  if (data.category) formData.append("category", data.category);
  formData.append("file", data.file);

  return apiFetch<{ id: string }>(`/knowledge/policies/${id}/upload`, {
    method: "PATCH",
    body: formData,
  });
}

export function createPolicy(data: { title: string; category: string; content: string }): Promise<{ id: string }> {
  return apiFetch<{ id: string }>("/knowledge/policies", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updatePolicy(id: string, data: { title?: string; category?: string; content?: string }): Promise<void> {
  return apiFetch<void>(`/knowledge/policies/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export function togglePolicy(id: string): Promise<{ isActive: boolean }> {
  return apiFetch<{ isActive: boolean }>(`/knowledge/policies/${id}/toggle`, {
    method: "POST",
  });
}

export function deletePolicy(id: string): Promise<void> {
  return apiFetch<void>(`/knowledge/policies/${id}`, {
    method: "DELETE",
  });
}

export interface FAQData {
  id: string;
  documentType: "faq";
  question: string;
  answer: string;
  preview: string;
  category: string;
  isActive: boolean;
  usageCount: number;
}

export function getFaqs(): Promise<FAQData[]> {
  return apiFetch<FAQData[]>("/knowledge/faqs");
}

export function uploadFaqDocument(data: { question?: string; category?: string; file: File }): Promise<{ id: string }> {
  const formData = new FormData();
  if (data.question) formData.append("question", data.question);
  if (data.category) formData.append("category", data.category);
  formData.append("file", data.file);

  return apiFetch<{ id: string }>("/knowledge/faqs/upload", {
    method: "POST",
    body: formData,
  });
}

export function replaceFaqDocument(id: string, data: { question?: string; category?: string; file: File }): Promise<{ id: string }> {
  const formData = new FormData();
  if (data.question) formData.append("question", data.question);
  if (data.category) formData.append("category", data.category);
  formData.append("file", data.file);

  return apiFetch<{ id: string }>(`/knowledge/faqs/${id}/upload`, {
    method: "PATCH",
    body: formData,
  });
}

export function createFaq(data: { question: string; answer: string; category: string }): Promise<{ id: string }> {
  return apiFetch<{ id: string }>("/knowledge/faqs", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateFaq(id: string, data: { question?: string; answer?: string; category?: string }): Promise<void> {
  return apiFetch<void>(`/knowledge/faqs/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export function toggleFaq(id: string): Promise<{ isActive: boolean }> {
  return apiFetch<{ isActive: boolean }>(`/knowledge/faqs/${id}/toggle`, {
    method: "POST",
  });
}

export function deleteFaq(id: string): Promise<void> {
  return apiFetch<void>(`/knowledge/faqs/${id}`, {
    method: "DELETE",
  });
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

export interface AgentPerformance {
  id: string;
  name: string;
  role: string;
}

export function getAgentPerformanceList(): Promise<AgentPerformance[]> {
  return apiFetch<AgentPerformance[]>("/agents");
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

export function getAssistantHistory(): Promise<AssistantResponse[]> {
  return apiFetch<AssistantResponse[]>("/assistant/history");
}

// ── Auth ──────────────────────────────────────────────────────────────────────

const API_BASE_ROOT = "http://localhost:8000/api/v1";

// Mock values for demonstration
const MOCK_MANAGER: User = {
  id: "b0000000-0000-0000-0000-000000000001",
  email: "manager@niletech.com",
  name: "Galal Manager",
  role: "manager",
  organization_id: "a0000000-0000-0000-0000-000000000001",
  is_active: true
};

const MOCK_AGENT: User = {
  id: "b0000001-0000-0000-0000-000000000002",
  email: "mohsen@niletech.com",
  name: "Mohsen Agent",
  role: "agent",
  agent_type: "human",
  organization_id: "a0000000-0000-0000-0000-000000000001",
  is_active: true
};

let currentUser: User | null = null;

export async function loginWithEmail(email: string, password: string): Promise<{ access_token: string }> {
  // Simple mock logic for demonstration
  if (password === "password") {
    if (email === "manager@niletech.com") {
      currentUser = MOCK_MANAGER;
      return { access_token: "mock-token-manager" };
    } else if (email === "mohsen@niletech.com") {
      currentUser = MOCK_AGENT;
      return { access_token: "mock-token-agent" };
    }
  }

  const formData = new URLSearchParams();
  formData.append("username", email);
  formData.append("password", password);

  try {
    const res = await fetch(`${API_BASE_ROOT}/auth/login/access-token`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: formData,
    });

    if (!res.ok) {
      throw new Error("Invalid email or password");
    }

    return res.json();
  } catch (err) {
    if (password === "password" && (email === "manager@niletech.com" || email === "mohsen@niletech.com")) {
       // Fallback to mock if backend is down
       return { access_token: "mock-token-fallback" };
    }
    throw err;
  }
}

export async function loginWithGoogle(idToken: string): Promise<{ access_token: string }> {
  return apiFetch<{ access_token: string }>(`/auth/google?token=${idToken}`, {
    method: "POST",
  });
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: "manager" | "agent";
  agent_type?: "human" | "ai" | null;
  organization_id: string;
  is_active: boolean;
}

export async function getUserMe(): Promise<User> {
  if (currentUser) return currentUser;
  return apiFetch<User>("/users/me");
}

export async function logoutUser(): Promise<void> {
  await apiFetch("/auth/logout", { method: "POST" });
}

