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

  it('shows cached llm insights without exposing a refresh action', () => {
    cy.visitAs('agent', '/agent/calls/int-001');
    cy.wait('@getInteractionDetail');
    cy.contains('Account Login').should('be.visible');

    cy.contains('LLM Coaching Insights').should('be.visible');
    cy.get('[data-cy="llm-refresh"]').should('not.exist');
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
