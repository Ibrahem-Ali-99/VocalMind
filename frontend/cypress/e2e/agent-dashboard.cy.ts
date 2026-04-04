describe('Agent Dashboard', () => {
  beforeEach(() => {
    cy.loginAs('agent');
    cy.wait('@getAgents');
    cy.wait('@getAgentProfile');
  });

  it('loads the default agent profile metrics', () => {
    cy.contains('Sarah M.').should('be.visible');
    cy.contains('Calls This Week').should('be.visible');
    cy.contains('Overall Score').should('be.visible');
    cy.contains('#1').should('be.visible');
    cy.contains('34').should('exist');
  });

  it('opens a recent call from the dashboard', () => {
    cy.get('a[href^="/agent/calls/"]').first().click();

    cy.wait('@getInteractionDetail');
    cy.location('pathname').should('match', /\/agent\/calls\/.+/);
    cy.contains('CALL DETAIL').should('exist');
  });
});
