describe('Session Detail', () => {
  it('renders resolved evaluation details from the api response', () => {
    cy.visitAs('manager', '/manager/inspector/int-001');

    cy.wait('@getInteractionDetail');
    cy.contains('Automated Evaluation').should('be.visible');
    cy.contains('Process Adherence').should('be.visible');
    cy.contains('Resolved').should('be.visible');
    cy.contains('Account Login').should('be.visible');
    cy.contains('9/10').should('be.visible');
  });

  it('renders missing sop steps and policy inference for unresolved sessions', () => {
    cy.visitAs('manager', '/manager/inspector/int-002');

    cy.wait('@getInteractionDetail');
    cy.contains('Unresolved').should('be.visible');
    cy.contains('Immediate empathy statement').should('be.visible');
    cy.contains('Supervisor escalation').should('be.visible');
    cy.contains('Policy Inference').scrollIntoView().should('be.visible');
    cy.contains('Contradiction').should('exist');
    cy.contains('Policy Version: 2025.03').should('exist');
    cy.contains('Conflict Resolved').should('exist');
  });

  it('opens and closes a transcript conversation window', () => {
    cy.visitAs('manager', '/manager/inspector/int-001');

    cy.wait('@getInteractionDetail');
    cy.get('[data-cy="transcript-window-trigger"]').first().click({
      force: true,
    });

    cy.get('[data-cy="conversation-window"]').should('be.visible');
    cy.contains('Conversation Window').should('be.visible');
    cy.get('[data-cy="transcript-window-close"]').click();
    cy.get('[data-cy="conversation-window"]').should('not.exist');
  });

  it('shows an error state when the session request fails', () => {
    cy.visitAs('manager', '/manager/inspector/int-999', {
      interactionDetails: {
        'int-999': {
          statusCode: 500,
        },
      },
    });

    cy.wait('@getInteractionDetail');
    cy.contains('Failed to load session').should('be.visible');
  });
});
