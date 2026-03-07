/**
 * Mock data for frontend development.
 * Replace with real API calls when backend endpoints are ready.
 */

// ── Manager Dashboard ────────────────────────────────────────────────────────

export const mockWeeklyTrend = [
  { day: "Mon", score: 82 },
  { day: "Tue", score: 85 },
  { day: "Wed", score: 79 },
  { day: "Thu", score: 88 },
  { day: "Fri", score: 84 },
  { day: "Sat", score: 90 },
  { day: "Sun", score: 86 },
];

export const mockEmotionDistribution = [
  { name: "Neutral", value: 40, color: "#6B7280" },
  { name: "Happy", value: 25, color: "#10B981" },
  { name: "Frustrated", value: 15, color: "#F59E0B" },
  { name: "Angry", value: 10, color: "#EF4444" },
  { name: "Sad", value: 10, color: "#3B82F6" },
];

export const mockPolicyCompliance = [
  { category: "Greeting Protocol", rate: 94, color: "#10B981" },
  { category: "Data Privacy", rate: 88, color: "#3B82F6" },
  { category: "Escalation Policy", rate: 76, color: "#F59E0B" },
  { category: "Hold Time Limit", rate: 82, color: "#8B5CF6" },
  { category: "Closing Script", rate: 91, color: "#10B981" },
];

export const mockAgentPerformance = [
  { name: "Sarah M.", empathy: 92, policy: 88, resolution: 85, overallScore: 88, trend: "up" as const },
  { name: "John D.", empathy: 78, policy: 95, resolution: 82, overallScore: 85, trend: "up" as const },
  { name: "Emily R.", empathy: 85, policy: 80, resolution: 90, overallScore: 85, trend: "down" as const },
  { name: "Mike T.", empathy: 72, policy: 85, resolution: 78, overallScore: 78, trend: "down" as const },
  { name: "Lisa K.", empathy: 88, policy: 76, resolution: 72, overallScore: 79, trend: "up" as const },
];

// ── Interactions ──────────────────────────────────────────────────────────────

export const mockInteractions = [
  {
    id: "int-001",
    agentName: "Sarah M.",
    agentId: "agent-001",
    date: "2025-03-01",
    time: "09:15 AM",
    duration: "8:42",
    language: "English",
    overallScore: 92,
    empathyScore: 95,
    policyScore: 90,
    resolutionScore: 88,
    resolved: true,
    hasViolation: false,
    hasOverlap: false,
    responseTime: "1.2s",
    status: "completed",
  },
  {
    id: "int-002",
    agentName: "John D.",
    agentId: "agent-002",
    date: "2025-03-01",
    time: "10:30 AM",
    duration: "12:15",
    language: "English",
    overallScore: 68,
    empathyScore: 62,
    policyScore: 70,
    resolutionScore: 55,
    resolved: false,
    hasViolation: true,
    hasOverlap: true,
    responseTime: "3.8s",
    status: "completed",
  },
  {
    id: "int-003",
    agentName: "Emily R.",
    agentId: "agent-003",
    date: "2025-03-01",
    time: "11:00 AM",
    duration: "6:20",
    language: "Arabic",
    overallScore: 85,
    empathyScore: 88,
    policyScore: 82,
    resolutionScore: 84,
    resolved: true,
    hasViolation: false,
    hasOverlap: false,
    responseTime: "1.5s",
    status: "completed",
  },
  {
    id: "int-004",
    agentName: "Mike T.",
    agentId: "agent-004",
    date: "2025-03-01",
    time: "02:45 PM",
    duration: "15:30",
    language: "English",
    overallScore: 55,
    empathyScore: 50,
    policyScore: 45,
    resolutionScore: 60,
    resolved: false,
    hasViolation: true,
    hasOverlap: false,
    responseTime: "5.1s",
    status: "completed",
  },
  {
    id: "int-005",
    agentName: "Lisa K.",
    agentId: "agent-005",
    date: "2025-03-01",
    time: "04:20 PM",
    duration: "4:15",
    language: "English",
    overallScore: 78,
    empathyScore: 75,
    policyScore: 82,
    resolutionScore: 78,
    resolved: true,
    hasViolation: false,
    hasOverlap: false,
    responseTime: "2.5s",
    status: "completed",
  },
];

// ── Utterances & Emotion Events ──────────────────────────────────────────────

export const mockUtterances = [
  {
    id: "utt-001",
    interactionId: "int-001",
    speaker: "agent",
    text: "Good morning! Thank you for calling VocalMind support. How may I help you today?",
    startTime: 10.5,
    endTime: 14.3,
    timestamp: "00:10",
    emotion: "neutral",
    confidence: 0.90,
  },
  {
    id: "utt-004",
    interactionId: "int-005",
    speaker: "customer",
    text: "Wow, that was fast! I'm so happy you resolved this so quickly.",
    startTime: 120.0,
    endTime: 125.0,
    timestamp: "02:00",
    emotion: "happy",
    confidence: 0.98,
  },
  {
    id: "utt-002",
    interactionId: "int-001",
    speaker: "customer",
    text: "Hi, I've been having issues with my account login for the past two days.",
    startTime: 5.5,
    endTime: 10.1,
    timestamp: "00:05",
    emotion: "frustrated",
    confidence: 0.85,
  },
  {
    id: "utt-003",
    interactionId: "int-001",
    speaker: "agent",
    text: "I'm sorry to hear that. Let me look into your account right away.",
    startTime: 10.5,
    endTime: 14.3,
    timestamp: "00:10",
    emotion: "empathetic",
    confidence: 0.88,
  },
];

export const mockEmotionEvents = [
  {
    id: "emo-001",
    interactionId: "int-001",
    previousEmotion: "neutral",
    newEmotion: "frustrated",
    fromEmotion: "neutral",
    toEmotion: "frustrated",
    jumpToSeconds: 5.5,
    timestamp: "00:05",
    confidenceScore: 0.85,
    delta: -0.3,
    speaker: "customer",
    llmJustification: "Customer expressed multi-day frustration with login issues.",
    justification: "Customer expressed multi-day frustration with login issues.",
  },
  {
    id: "emo-002",
    interactionId: "int-002",
    previousEmotion: "frustrated",
    newEmotion: "angry",
    fromEmotion: "frustrated",
    toEmotion: "angry",
    jumpToSeconds: 45.0,
    timestamp: "00:45",
    confidenceScore: 0.91,
    delta: -0.5,
    speaker: "customer",
    llmJustification: "Customer raised voice after being put on hold for the third time.",
    justification: "Customer raised voice after being put on hold for the third time.",
  },
];

export const mockPolicyViolations = [
  {
    id: "vio-001",
    interactionId: "int-002",
    policyName: "Hold Time Limit",
    policyTitle: "Hold Time Limit",
    category: "Process",
    description: "Customer was placed on hold for over 3 minutes without check-in.",
    reasoning: "Agent exceeded the 2-minute hold policy without providing an update to the customer.",
    severity: "high",
    score: 45,
    timestamp: "02:00",
  },
  {
    id: "vio-002",
    interactionId: "int-004",
    policyName: "Escalation Policy",
    policyTitle: "Escalation Policy",
    category: "Process",
    description: "Agent failed to escalate after customer requested a supervisor twice.",
    reasoning: "Customer explicitly requested supervisor at 03:20 and 04:15, agent did not initiate transfer.",
    severity: "critical",
    score: 30,
    timestamp: "03:45",
  },
];

// ── Agent Data ───────────────────────────────────────────────────────────────

export const mockAgentPersonalData = {
  id: "agent-001",
  name: "Sarah M.",
  role: "Senior Agent",
  totalCalls: 156,
  avgScore: 88,
  overallScore: 88,
  callsThisWeek: 34,
  teamRank: 1,
  empathyScore: 92,
  policyScore: 88,
  resolutionScore: 85,
  resolutionRate: 91,
  avgResponseTime: "1.4s",
  trend: "up" as const,
  weeklyTrend: [
    { day: "Mon", score: 85 },
    { day: "Tue", score: 88 },
    { day: "Wed", score: 82 },
    { day: "Thu", score: 91 },
    { day: "Fri", score: 90 },
  ],
  recentCalls: [
    { id: "int-001", date: "2025-03-01", time: "09:15 AM", score: 92, duration: "8:42", language: "English", resolved: true, hasReview: true },
    { id: "int-005", date: "2025-02-28", time: "02:30 PM", score: 85, duration: "11:20", language: "English", resolved: true, hasReview: false },
    { id: "int-006", date: "2025-02-28", time: "04:10 PM", score: 78, duration: "14:05", language: "Arabic", resolved: false, hasReview: true },
  ],
};

// ── Knowledge Base ───────────────────────────────────────────────────────────

export const mockPolicies = [
  {
    id: "pol-001",
    title: "Greeting Protocol",
    category: "Communication",
    content: "All agents must greet customers within the first 5 seconds of the call with the standard greeting script.",
    preview: "All agents must greet customers within the first 5 seconds...",
    lastUpdated: "2025-02-15",
    isActive: true,
  },
  {
    id: "pol-002",
    title: "Data Privacy Guidelines",
    category: "Security",
    content: "Never request full credit card numbers over the phone. Use the secure verification portal instead.",
    preview: "Never request full credit card numbers over the phone...",
    lastUpdated: "2025-01-20",
    isActive: true,
  },
  {
    id: "pol-003",
    title: "Escalation Procedure",
    category: "Process",
    content: "If a customer requests a supervisor, transfer within 60 seconds. Document the reason for escalation.",
    preview: "If a customer requests a supervisor, transfer within 60 seconds...",
    lastUpdated: "2025-02-01",
    isActive: false,
  },
];

export const mockFAQs = [
  {
    id: "faq-001",
    question: "How do I reset a customer's password?",
    answer: "Navigate to Admin > User Management > Search for the customer > Click Reset Password.",
    preview: "Navigate to Admin > User Management > Search...",
    category: "Account Management",
    isActive: true,
  },
  {
    id: "faq-002",
    question: "What is the refund policy?",
    answer: "Customers are eligible for a full refund within 30 days of purchase. After 30 days, a 15% restocking fee applies.",
    preview: "Customers are eligible for a full refund within 30 days...",
    category: "Billing",
    isActive: true,
  },
];

// ── Manager Assistant ────────────────────────────────────────────────────────

export const mockAssistantMessages: Array<{
  id: string;
  type: "assistant" | "user" | "ai";
  content: string;
  timestamp?: string;
  mode?: string;
  sql?: string;
  executionTime?: string;
}> = [
  {
    id: "msg-001",
    type: "assistant",
    content: "Hello! I'm your VocalMind assistant. I can help you analyze agent performance, review policy compliance, and answer questions about your team's metrics. What would you like to know?",
    timestamp: "09:00 AM",
  },
];
