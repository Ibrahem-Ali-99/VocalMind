import {
  mockAgentPerformance,
  mockAgentPersonalData,
  mockEmotionDistribution,
  mockEmotionEvents,
  mockFAQs,
  mockInteractions,
  mockPolicies,
  mockPolicyCompliance,
  mockPolicyViolations,
  mockUtterances,
  mockWeeklyTrend,
  mockKBArticles,
} from '../../src/app/data/mockData';
import type {
  AgentProfile,
  AgentSummary,
  AssistantResponse,
  DashboardStats,
  EmotionEventData,
  FAQData,
  InteractionDetail,
  InteractionSummary,
  PolicyData,
  PolicyViolationData,
  User,
  UtteranceData,
} from '../../src/app/services/api';

export type TestRole = 'manager' | 'agent';

export interface MockRouteResponse<T> {
  body?: T;
  delayMs?: number;
  forceNetworkError?: boolean;
  statusCode?: number;
}

export interface AppScenario {
  agentProfile?: MockRouteResponse<AgentProfile>;
  agents?: MockRouteResponse<AgentSummary[]>;
  assistantHistory?: MockRouteResponse<AssistantResponse[]>;
  assistantQuery?: MockRouteResponse<AssistantResponse>;
  auth?: { role: TestRole } | false;
  chat?: MockRouteResponse<{ answer: string; context: string }>;
  dashboard?: MockRouteResponse<DashboardStats>;
  faqs?: MockRouteResponse<FAQData[]>;
  kb?: MockRouteResponse<FAQData[]>;
  interactionDetails?: Record<string, MockRouteResponse<InteractionDetail>>;
  interactions?: MockRouteResponse<InteractionSummary[]>;
  logout?: MockRouteResponse<Record<string, never>>;
  policies?: MockRouteResponse<PolicyData[]>;
}

const baseInteraction: InteractionSummary = {
  id: 'int-001',
  agentId: 'agent-001',
  agentName: 'Sarah M.',
  audioFilePath: '/audio/int-001.mp3',
  date: '2025-03-01',
  duration: '8:42',
  empathyScore: 95,
  hasOverlap: false,
  hasViolation: false,
  language: 'English',
  overallScore: 92,
  policyScore: 90,
  resolved: true,
  resolutionScore: 88,
  responseTime: '1.2s',
  status: 'completed',
  time: '09:15 AM',
};

const defaultInteractions = mockInteractions.map((interaction) =>
  buildInteractionSummary({
    ...interaction,
    audioFilePath: `/audio/${interaction.id}.mp3`,
  }),
);

export function buildInteractionSummary(
  overrides: Partial<InteractionSummary> = {},
): InteractionSummary {
  const id = overrides.id ?? baseInteraction.id;

  return {
    ...baseInteraction,
    ...overrides,
    audioFilePath:
      overrides.audioFilePath === undefined
        ? `/audio/${id}.mp3`
        : overrides.audioFilePath,
    id,
  };
}

export function buildDashboardStats(
  overrides: Partial<DashboardStats> = {},
): DashboardStats {
  return {
    kpis: {
      avgScore: 88,
      totalCalls: 342,
      resolutionRate: 91,
      violationCount: 12,
    },
    weeklyTrend: cloneData(mockWeeklyTrend),
    emotionDistribution: cloneData(mockEmotionDistribution),
    policyCompliance: cloneData(mockPolicyCompliance),
    agentPerformance: cloneData(mockAgentPerformance),
    interactions: cloneData(defaultInteractions),
    ...overrides,
  };
}

export function buildAssistantResponse(
  overrides: Partial<AssistantResponse> = {},
): AssistantResponse {
  return {
    id: 'resp-001',
    type: 'ai',
    content: "I've analyzed your query. Finding top performing agents...",
    mode: 'chat',
    sql: 'SELECT * FROM interactions LIMIT 5',
    execution_time: '120ms',
    executionTime: '120ms',
    data: [
      { name: 'Sarah M.', score: 92 },
      { name: 'John D.', score: 85 },
    ],
    success: true,
    ...overrides,
  };
}

export function buildAgentProfile(
  overrides: Partial<AgentProfile> = {},
): AgentProfile {
  return {
    ...cloneData(mockAgentPersonalData),
    ...overrides,
  };
}

function buildUser(role: TestRole): User {
  if (role === 'agent') {
    return {
      id: 'usr-002',
      email: 'agent@vocalmind.ai',
      name: 'Robert King',
      role: 'agent',
      agent_type: 'human',
      organization_id: 'org-001',
      is_active: true,
    };
  }

  return {
    id: 'usr-001',
    email: 'manager@vocalmind.ai',
    name: 'Manager King',
    role: 'manager',
    organization_id: 'org-001',
    is_active: true,
  };
}

function buildUtterances(interaction: InteractionSummary): UtteranceData[] {
  const cannedUtterances = cloneData(
    mockUtterances.filter((utterance) => utterance.interactionId === interaction.id),
  );

  if (cannedUtterances.length > 0) {
    return cannedUtterances.map((utterance) => ({
      ...utterance,
      fusedEmotion: (utterance as any).fusedEmotion ?? utterance.emotion,
      fusedConfidence: (utterance as any).fusedConfidence ?? utterance.confidence,
    }));
  }

  const prefix = interaction.id.replace('int-', 'utt-');

  if (interaction.resolved) {
    return [
      {
        id: `${prefix}-001`,
        interactionId: interaction.id,
        speaker: 'customer',
        text: "Hi, I've been having issues with my account login for the past two days.",
        startTime: 5,
        endTime: 10,
        timestamp: '00:05',
        emotion: 'frustrated',
        confidence: 0.86,
        fusedEmotion: 'frustrated',
        fusedConfidence: 0.87,
      },
      {
        id: `${prefix}-002`,
        interactionId: interaction.id,
        speaker: 'agent',
        text: 'I am sorry to hear that. Let me look into your account right away.',
        startTime: 10,
        endTime: 15,
        timestamp: '00:10',
        emotion: 'empathetic',
        confidence: 0.91,
        fusedEmotion: 'empathetic',
        fusedConfidence: 0.9,
      },
      {
        id: `${prefix}-003`,
        interactionId: interaction.id,
        speaker: 'customer',
        text: 'Thank you, I just need to get back into the portal before my shift starts.',
        startTime: 16,
        endTime: 24,
        timestamp: '00:16',
        emotion: 'curious',
        confidence: 0.74,
        fusedEmotion: 'interested',
        fusedConfidence: 0.78,
      },
      {
        id: `${prefix}-004`,
        interactionId: interaction.id,
        speaker: 'agent',
        text: 'You are all set now. Please try again and let me know if you can sign in.',
        startTime: 25,
        endTime: 34,
        timestamp: '00:25',
        emotion: 'helpful',
        confidence: 0.88,
        fusedEmotion: 'helpful',
        fusedConfidence: 0.89,
      },
    ];
  }

  return [
    {
      id: `${prefix}-001`,
      interactionId: interaction.id,
      speaker: 'customer',
      text: 'I have already explained this twice. I still have not received my refund.',
      startTime: 4,
      endTime: 11,
      timestamp: '00:04',
      emotion: 'frustrated',
      confidence: 0.9,
      fusedEmotion: 'frustrated',
      fusedConfidence: 0.92,
    },
    {
      id: `${prefix}-002`,
      interactionId: interaction.id,
      speaker: 'agent',
      text: 'Let me review the case. Please hold while I check the notes.',
      startTime: 12,
      endTime: 18,
      timestamp: '00:12',
      emotion: 'neutral',
      confidence: 0.7,
      fusedEmotion: 'professional',
      fusedConfidence: 0.73,
    },
    {
      id: `${prefix}-003`,
      interactionId: interaction.id,
      speaker: 'customer',
      text: 'This is getting ridiculous. I asked for a supervisor already.',
      startTime: 42,
      endTime: 49,
      timestamp: '00:42',
      emotion: 'angry',
      confidence: 0.96,
      fusedEmotion: 'angry',
      fusedConfidence: 0.97,
    },
    {
      id: `${prefix}-004`,
      interactionId: interaction.id,
      speaker: 'agent',
      text: 'I understand. I will escalate this now and stay with you on the line.',
      startTime: 50,
      endTime: 58,
      timestamp: '00:50',
      emotion: 'neutral',
      confidence: 0.69,
      fusedEmotion: 'neutral',
      fusedConfidence: 0.7,
    },
  ];
}

function buildEmotionEvents(interaction: InteractionSummary): EmotionEventData[] {
  const cannedEvents = cloneData(
    mockEmotionEvents.filter((event) => event.interactionId === interaction.id),
  );

  if (cannedEvents.length > 0) {
    return cannedEvents;
  }

  const prefix = interaction.id.replace('int-', 'emo-');

  if (interaction.resolved) {
    return [
      {
        id: `${prefix}-001`,
        interactionId: interaction.id,
        previousEmotion: 'neutral',
        newEmotion: 'frustrated',
        fromEmotion: 'neutral',
        toEmotion: 'frustrated',
        jumpToSeconds: 5,
        timestamp: '00:05',
        confidenceScore: 0.85,
        delta: -0.3,
        speaker: 'customer',
        llmJustification: 'Customer described multi-day login issues.',
        justification: 'Customer described multi-day login issues.',
      },
      {
        id: `${prefix}-002`,
        interactionId: interaction.id,
        previousEmotion: 'frustrated',
        newEmotion: 'calmer',
        fromEmotion: 'frustrated',
        toEmotion: 'calmer',
        jumpToSeconds: 25,
        timestamp: '00:25',
        confidenceScore: 0.82,
        delta: 0.35,
        speaker: 'customer',
        llmJustification: 'Customer calmed once the reset steps were confirmed.',
        justification: 'Customer calmed once the reset steps were confirmed.',
      },
    ];
  }

  return [
    {
      id: `${prefix}-001`,
      interactionId: interaction.id,
      previousEmotion: 'frustrated',
      newEmotion: 'angry',
      fromEmotion: 'frustrated',
      toEmotion: 'angry',
      jumpToSeconds: 42,
      timestamp: '00:42',
      confidenceScore: 0.91,
      delta: -0.5,
      speaker: 'customer',
      llmJustification: 'Customer escalated after repeated delays.',
      justification: 'Customer escalated after repeated delays.',
    },
  ];
}

function buildPolicyViolations(
  interaction: InteractionSummary,
): PolicyViolationData[] {
  const cannedViolations = cloneData(
    mockPolicyViolations.filter((violation) => violation.interactionId === interaction.id),
  );

  if (cannedViolations.length > 0) {
    return cannedViolations;
  }

  if (!interaction.hasViolation) {
    return [];
  }

  return [
    {
      id: `vio-${interaction.id}`,
      interactionId: interaction.id,
      policyName: interaction.resolved ? 'Closing Script' : 'Escalation Policy',
      policyTitle:
        interaction.resolved ? 'Closing Script' : 'Escalation Policy',
      category: interaction.resolved ? 'Communication' : 'Process',
      description: interaction.resolved
        ? 'Agent did not confirm whether additional help was needed.'
        : 'Agent failed to escalate after the customer requested a supervisor.',
      reasoning: interaction.resolved
        ? 'Agent closed the call without the required closing confirmation.'
        : 'Customer requested escalation twice and no transfer was initiated.',
      severity: interaction.resolved ? 'medium' : 'critical',
      score: interaction.resolved ? 72 : 38,
      timestamp: interaction.resolved ? '07:55' : '03:45',
    },
  ];
}

function buildLlmTriggers(
  interaction: InteractionSummary,
  policyViolations: PolicyViolationData[],
) {
  const unresolved = !interaction.resolved;

  return {
    available: true,
    interactionId: interaction.id,
    emotionShift: {
      isDissonanceDetected: unresolved,
      dissonanceType: unresolved ? 'Late Empathy' : 'None',
      rootCause: unresolved
        ? 'Customer frustration escalated before empathy or escalation language appeared.'
        : 'Agent responded promptly and reduced customer frustration.',
      currentCustomerEmotion: unresolved ? 'angry' : 'calmer',
      currentEmotionReasoning: unresolved
        ? 'Customer repeated the complaint and requested a supervisor.'
        : 'Customer acknowledged the fix and the tone softened.',
      counterfactualCorrection: unresolved
        ? 'Acknowledge the frustration immediately and offer escalation sooner.'
        : 'Continue confirming resolution and next steps early in the call.',
      evidenceQuotes: [],
      citations: [],
    },
    processAdherence: {
      detectedTopic: unresolved ? 'Refund follow-up' : 'Account Login',
      isResolved: interaction.resolved,
      efficiencyScore: unresolved ? 5 : 9,
      justification: unresolved
        ? 'Agent delayed escalation and missed the required supervisor transfer step.'
        : 'Agent followed the login recovery SOP and confirmed the fix.',
      missingSopSteps: unresolved
        ? ['Immediate empathy statement', 'Supervisor escalation']
        : [],
      evidenceQuotes: [],
      citations: [],
    },
    nliPolicy: {
      nliCategory: (policyViolations.length > 0 ? 'Contradiction' : 'Entailment') as "Contradiction" | "Entailment" | "Benign Deviation" | "Policy Hallucination",
      justification:
        policyViolations.length > 0
          ? 'The transcript conflicts with the escalation policy requirements.'
          : 'The transcript aligns with the documented policy expectations.',
      evidenceQuotes: [],
      citations: [],
      policyVersion: '2025.03',
      policyEffectiveAt: '2025-03-01',
      policyCategory: policyViolations.length > 0 ? 'Process' : 'Support',
      conflictResolutionApplied: policyViolations.length > 0,
    },
    derived: {
      customerText: unresolved
        ? 'I asked for a supervisor already.'
        : 'Thank you, I can sign in now.',
      acousticEmotion: unresolved ? 'angry' : 'calmer',
      fusedEmotion: unresolved ? 'angry' : 'calmer',
      agentStatement: unresolved
        ? 'Please hold while I check the notes.'
        : 'You are all set now.',
    },
  };
}

export function buildInteractionDetail(
  interaction: InteractionSummary = defaultInteractions[0],
  overrides: Partial<InteractionDetail> = {},
): InteractionDetail {
  const policyViolations = buildPolicyViolations(interaction);
  const llmTriggers = buildLlmTriggers(interaction, policyViolations);

  return {
    emotionComparison: null,
    ragCompliance: null,
    emotionTriggers: null,
    ...overrides,
    interaction: {
      ...interaction,
      ...(overrides.interaction ?? {}),
    },
    utterances: overrides.utterances ?? buildUtterances(interaction),
    emotionEvents: overrides.emotionEvents ?? buildEmotionEvents(interaction),
    policyViolations: overrides.policyViolations ?? policyViolations,
    llmTriggers: overrides.llmTriggers ?? llmTriggers,
  };
}

function buildDefaultInteractionDetailMap(
  interactions: InteractionSummary[],
): Record<string, MockRouteResponse<InteractionDetail>> {
  return Object.fromEntries(
    interactions.map((interaction) => [
      interaction.id,
      {
        body: buildInteractionDetail(interaction),
      },
    ]),
  );
}

function buildAgentList(interactions: InteractionSummary[]): AgentSummary[] {
  const agentMap = new Map<string, AgentSummary>();

  interactions.forEach((interaction) => {
    if (!agentMap.has(interaction.agentId)) {
      agentMap.set(interaction.agentId, {
        id: interaction.agentId,
        name: interaction.agentName,
        role: 'Agent',
      });
    }
  });

  return [...agentMap.values()];
}

function cloneData<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function inferRoleFromLoginBody(body: unknown): TestRole {
  if (typeof body === 'string') {
    const username = new URLSearchParams(body).get('username') ?? '';
    return username.includes('mohsen') || username.includes('agent')
      ? 'agent'
      : 'manager';
  }

  if (typeof body === 'object' && body !== null && 'username' in body) {
    const username = String((body as Record<string, unknown>).username ?? '');
    return username.includes('mohsen') || username.includes('agent')
      ? 'agent'
      : 'manager';
  }

  return 'manager';
}

function replyWith<T>(
  req: any,
  response: MockRouteResponse<T> | undefined,
  fallbackBody: T,
  alias?: string,
) {
  if (alias) {
    req.alias = alias;
  }

  if (response?.forceNetworkError) {
    req.reply({ forceNetworkError: true });
    return;
  }

  req.reply({
    statusCode: response?.statusCode ?? 200,
    delay: response?.delayMs,
    body: cloneData(response?.body ?? fallbackBody),
  });
}

export function registerApiScenario(scenario: AppScenario = {}) {
  const initialRole =
    scenario.auth === false ? null : scenario.auth?.role ?? 'manager';
  const authState: { role: TestRole | null } = { role: initialRole };

  const interactions = cloneData(
    scenario.interactions?.body ?? defaultInteractions,
  );
  const dashboard = buildDashboardStats({
    interactions,
  });
  const interactionDetails = {
    ...buildDefaultInteractionDetailMap(interactions),
    ...(scenario.interactionDetails ?? {}),
  };
  const agents = buildAgentList(interactions);
  const agentProfile = buildAgentProfile();

  cy.intercept('GET', '**/api/v1/dashboard/stats', (req) => {
    replyWith(req, scenario.dashboard, dashboard, 'getDashboardStats');
  });

  cy.intercept('GET', '**/api/v1/interactions', (req) => {
    replyWith(req, scenario.interactions, interactions, 'getInteractions');
  });

  cy.intercept('GET', '**/api/v1/interactions/*', (req) => {
    const segments = req.url.split('/');
    const idWithQuery = segments[segments.length - 1] ?? '';
    const interactionId = idWithQuery.split('?')[0];
    const interaction =
      interactions.find((item) => item.id === interactionId) ??
      buildInteractionSummary({
        id: interactionId,
        agentId: 'agent-999',
        agentName: 'Fallback Agent',
      });
    const response =
      interactionDetails[interactionId] ?? {
        body: buildInteractionDetail(interaction),
      };

    replyWith(
      req,
      response,
      buildInteractionDetail(interaction),
      'getInteractionDetail',
    );
  });

  cy.intercept('GET', '**/api/v1/knowledge/policies', (req) => {
    replyWith(req, scenario.policies, cloneData(mockPolicies), 'getPolicies');
  });

  cy.intercept('GET', '**/api/v1/knowledge/faqs', (req) => {
    replyWith(req, scenario.faqs, cloneData(mockFAQs), 'getFaqs');
  });

  cy.intercept('GET', '**/api/v1/knowledge/kb', (req) => {
    replyWith(req, scenario.kb, cloneData(mockKBArticles), 'getKB');
  });

  cy.intercept('GET', '**/api/v1/agents', (req) => {
    replyWith(req, scenario.agents, agents, 'getAgents');
  });

  cy.intercept('GET', '**/api/v1/agents/*', (req) => {
    replyWith(req, scenario.agentProfile, agentProfile, 'getAgentProfile');
  });

  cy.intercept('GET', '**/api/v1/assistant/history', (req) => {
    replyWith(req, scenario.assistantHistory, [], 'getAssistantHistory');
  });

  cy.intercept('POST', '**/api/v1/assistant/query', (req) => {
    replyWith(
      req,
      scenario.assistantQuery,
      buildAssistantResponse(),
      'postAssistantQuery',
    );
  });

  cy.intercept('POST', '**/api/v1/chat', (req) => {
    replyWith(
      req,
      scenario.chat,
      { answer: 'Here is your requested answer.', context: 'Knowledge context' },
      'postChat',
    );
  });

  cy.intercept('GET', '**/api/v1/users/me', (req) => {
    req.alias = 'getUserMe';

    if (!authState.role) {
      req.reply({
        statusCode: 401,
        body: { detail: 'Unauthorized' },
      });
      return;
    }

    req.reply({
      statusCode: 200,
      body: buildUser(authState.role),
    });
  });

  cy.intercept('POST', '**/api/v1/auth/login/access-token', (req) => {
    authState.role = inferRoleFromLoginBody(req.body);
    req.alias = 'login';
    req.reply({
      statusCode: 200,
      body: {
        access_token: `mock-token-${authState.role}`,
        token_type: 'bearer',
      },
    });
  });

  cy.intercept('POST', '**/api/v1/auth/logout', (req) => {
    authState.role = null;
    replyWith(req, scenario.logout, {}, 'logout');
  });
}
