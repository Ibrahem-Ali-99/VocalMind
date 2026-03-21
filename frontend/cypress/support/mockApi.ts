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
        policyViolations: mockPolicyViolations,
        llmTriggers: { 
          available: true, 
          processAdherence: { isResolved: true, detectedTopic: "Account Login", efficiencyScore: 9, justification: "Agent followed all steps.", missingSopSteps: [] },
          nliPolicy: { nliCategory: "Entailment", justification: "Agent followed policy.", evidenceQuotes: [] },
          emotionShift: { isDissonanceDetected: false, dissonanceType: "None", rootCause: "Professional", counterfactualCorrection: "", evidenceQuotes: [] }
        }
      }
    });
  });

  cy.intercept('GET', '**/api/v1/interactions/int-001*', {
    statusCode: 200,
    body: {
      interaction: mockInteractions[0],
      utterances: mockUtterances,
      emotionEvents: mockEmotionEvents,
      policyViolations: mockPolicyViolations,
      llmTriggers: { 
        available: true, 
        processAdherence: { isResolved: true, detectedTopic: "Account Login", efficiencyScore: 9, justification: "Agent followed all steps.", missingSopSteps: [] },
        nliPolicy: { nliCategory: "Entailment", justification: "Agent followed policy.", evidenceQuotes: [] },
        emotionShift: { isDissonanceDetected: false, dissonanceType: "None", rootCause: "Professional", counterfactualCorrection: "", evidenceQuotes: [] }
      }
    }
  }).as('getInteraction001');

  cy.intercept('GET', '**/api/v1/interactions/int-002*', {
    statusCode: 200,
    body: {
      interaction: mockInteractions[1],
      utterances: mockUtterances,
      emotionEvents: mockEmotionEvents,
      policyViolations: mockPolicyViolations,
      llmTriggers: { 
        available: true, 
        processAdherence: { isResolved: false, detectedTopic: "Billing Inquiry", efficiencyScore: 5, justification: "Agent missed closing statement.", missingSopSteps: ["Closing Statement"] },
        nliPolicy: { nliCategory: "Benign Deviation", justification: "Agent deviation was minor.", evidenceQuotes: [] },
        emotionShift: { isDissonanceDetected: true, dissonanceType: "Late Empathy", rootCause: "Agent delayed empathy", counterfactualCorrection: "Express empathy earlier", evidenceQuotes: [] }
      }
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

  cy.intercept('GET', '**/api/v1/assistant/history', {
    statusCode: 200,
    body: []
  }).as('getAssistantHistory');

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
