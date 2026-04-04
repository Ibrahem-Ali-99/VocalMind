describe('Manager Assistant', () => {
  it('submits a typed question and renders the response data', () => {
    cy.visitAs('manager', '/manager/assistant', {
      assistantQuery: {
        delayMs: 400,
      },
    });
    cy.wait('@getAssistantHistory');

    cy.get('input[placeholder*="Ask about scores"]')
      .type('What is the average score?{enter}');

    cy.contains('What is the average score?').should('be.visible');
    cy.get('[data-cy="assistant-loading"]').should('be.visible');
    cy.wait('@postAssistantQuery');

    cy.get('[data-cy="assistant-loading"]').should('not.exist');
    cy.contains("I've analyzed your query.").should('be.visible');
    cy.contains('Sarah M.').should('be.visible');
    cy.contains('John D.').should('be.visible');
  });

  it('submits a suggested query and exposes the generated sql details', () => {
    cy.visitAs('manager', '/manager/assistant');
    cy.wait('@getAssistantHistory');

    cy.get('[data-cy="assistant-suggestion-1"]').click();

    cy.wait('@postAssistantQuery');
    cy.contains('List all policy violations').should('be.visible');
    cy.contains('Show generated SQL').click();
    cy.contains('SELECT * FROM interactions LIMIT 5').should('be.visible');
    cy.contains('Executed in 120ms').should('be.visible');
  });

  it('shows a fallback message when the assistant request fails', () => {
    cy.visitAs('manager', '/manager/assistant', {
      assistantQuery: {
        statusCode: 500,
      },
    });
    cy.wait('@getAssistantHistory');

    cy.get('input[placeholder*="Ask about scores"]')
      .type('Which calls need review?{enter}');

    cy.wait('@postAssistantQuery');
    cy.contains(
      "I'm sorry, I'm having trouble connecting to the service.",
    ).should('be.visible');
  });
});
