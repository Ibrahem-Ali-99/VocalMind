import {
  buildInteractionDetail,
  buildInteractionSummary,
} from '../support/mockApi';

describe('Agent Call Detail', () => {
  it('renders coaching and llm insights for an unresolved call', () => {
    cy.visitAs('agent', '/agent/calls/int-002');

    cy.wait('@getInteractionDetail');
    cy.contains('Coaching Points').should('be.visible');
    cy.contains(/Hold Time Limit|Escalation Policy/).should('be.visible');
    cy.contains('LLM Coaching Insights').scrollIntoView().should('be.visible');
    cy.contains('Needs follow-up').should('exist');
    cy.contains('Contradiction').should('exist');
  });

  it('refreshes the llm insights with a rerun request', () => {
    const refreshedDetail = buildInteractionDetail(
      buildInteractionSummary({
        id: 'int-001',
        agentId: 'agent-001',
        agentName: 'Sarah M.',
      }),
      {
        llmTriggers: {
          available: true,
          interactionId: 'int-001',
          processAdherence: {
            detectedTopic: 'Refund follow-up',
            isResolved: false,
            efficiencyScore: 6,
            justification: 'Updated rerun justification.',
            missingSopSteps: ['Supervisor escalation'],
            evidenceQuotes: [],
            citations: [],
          },
          nliPolicy: {
            nliCategory: 'Benign Deviation',
            justification: 'Updated policy view.',
            evidenceQuotes: [],
            citations: [],
          },
          emotionShift: {
            isDissonanceDetected: false,
            dissonanceType: 'None',
            rootCause: 'Updated emotion analysis.',
            currentCustomerEmotion: 'calmer',
            currentEmotionReasoning: 'Customer accepted the next step.',
            counterfactualCorrection: 'Maintain current pacing.',
            evidenceQuotes: [],
            citations: [],
          },
        },
      },
    );

    cy.visitAs('agent', '/agent/calls/int-001');
    cy.wait('@getInteractionDetail');
    cy.contains('Account Login').should('be.visible');

    cy.intercept(
      'GET',
      '**/api/v1/interactions/int-001*llm_force_rerun=true*',
      {
        statusCode: 200,
        body: refreshedDetail,
      },
    ).as('refreshLLM');

    cy.get('[data-cy="llm-refresh"]').click();

    cy.wait('@refreshLLM');
    cy.contains('Refund follow-up').should('be.visible');
    cy.contains('Updated policy view.').should('be.visible');
  });

  it('shows an unavailable state when llm insights are offline', () => {
    cy.visitAs('agent', '/agent/calls/int-001', {
      interactionDetails: {
        'int-001': {
          body: buildInteractionDetail(buildInteractionSummary(), {
            llmTriggers: {
              available: false,
              error: 'LLM offline',
              interactionId: 'int-001',
            } as any,
          }),
        },
      },
    });

    cy.wait('@getInteractionDetail');
    cy.contains('LLM coaching insights unavailable.').should('be.visible');
    cy.contains('LLM offline').should('be.visible');
  });

  it('shows an error state when the call detail request fails', () => {
    cy.visitAs('agent', '/agent/calls/int-500', {
      interactionDetails: {
        'int-500': {
          statusCode: 500,
        },
      },
    });

    cy.wait('@getInteractionDetail');
    cy.contains('Failed to load call details').should('be.visible');
  });
});
