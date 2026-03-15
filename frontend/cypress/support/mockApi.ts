import { 
  mockWeeklyTrend, 
  mockEmotionDistribution, 
  mockPolicyCompliance, 
  mockAgentPerformance, 
  mockInteractions, 
  mockUtterances, 
  mockEmotionEvents, 
  mockPolicyViolations, 
  mockPolicies, 
  mockFAQs,
  mockAgentPersonalData 
} from '../../src/app/data/mockData';

beforeEach(() => {
  cy.intercept('GET', '**/api/v1/dashboard/stats', {
    statusCode: 200,
    body: {
      kpis: {
        avgScore: 88,
        totalCalls: 342,
        resolutionRate: 91,
        violationCount: 12
      },
      weeklyTrend: mockWeeklyTrend,
      emotionDistribution: mockEmotionDistribution,
      policyCompliance: mockPolicyCompliance,
      agentPerformance: mockAgentPerformance,
      interactions: mockInteractions
    }
  }).as('getDashboardStats');

  cy.intercept('GET', '**/api/v1/interactions', {
    statusCode: 200,
    body: mockInteractions
  }).as('getInteractions');

  // Wildcard fallback for any interaction detail (registered first so specific ones override via LIFO)
  cy.intercept('GET', '**/api/v1/interactions/int-*', (req) => {
    req.reply({
      statusCode: 200,
      body: {
        interaction: mockInteractions[0],
        utterances: mockUtterances,
        emotionEvents: mockEmotionEvents,
        policyViolations: mockPolicyViolations
      }
    });
  });

  cy.intercept('GET', '**/api/v1/interactions/int-001', {
    statusCode: 200,
    body: {
      interaction: mockInteractions[0],
      utterances: mockUtterances,
      emotionEvents: mockEmotionEvents,
      policyViolations: mockPolicyViolations
    }
  }).as('getInteraction001');

  cy.intercept('GET', '**/api/v1/interactions/int-002', {
    statusCode: 200,
    body: {
      interaction: mockInteractions[1],
      utterances: mockUtterances,
      emotionEvents: mockEmotionEvents,
      policyViolations: mockPolicyViolations
    }
  }).as('getInteraction002');

  cy.intercept('GET', '**/api/v1/knowledge/policies', {
    statusCode: 200,
    body: mockPolicies
  }).as('getPolicies');

  cy.intercept('GET', '**/api/v1/knowledge/faqs', {
    statusCode: 200,
    body: mockFAQs
  }).as('getFaqs');

  cy.intercept('GET', '**/api/v1/agents', {
    statusCode: 200,
    body: mockAgentPerformance.map((a, i) => ({ id: `agent-00${i+1}`, name: a.name, role: "Agent" }))
  }).as('getAgents');

  cy.intercept('GET', '**/api/v1/agents/*', {
    statusCode: 200,
    body: mockAgentPersonalData
  }).as('getAgentProfile');

  cy.intercept('POST', '**/api/v1/assistant/query', {
    statusCode: 200,
    body: {
      id: "resp-001",
      type: "ai",
      content: "I've analyzed your query. Finding top performing agents...",
      mode: "chat",
      sql: "SELECT * FROM interactions LIMIT 5",
      execution_time: "120ms",
      data: [
        { name: "Sarah M.", score: 92 },
        { name: "John D.", score: 85 }
      ],
      success: true
    }
  }).as('postAssistantQuery');

  cy.intercept('POST', '**/api/v1/chat', {
    statusCode: 200,
    body: { answer: "Here is your requested answer.", context: "Knowledge context" }
  }).as('postChat');
});
