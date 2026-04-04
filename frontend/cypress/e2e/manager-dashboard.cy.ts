describe('Manager Dashboard', () => {
  beforeEach(() => {
    cy.loginAs('manager');
    cy.wait('@getDashboardStats');
  });

  it('renders KPI cards from the dashboard response', () => {
    cy.contains('Average Score').should('be.visible');
    cy.contains('88%').should('be.visible');
    cy.contains('Calls Processed').should('be.visible');
    cy.contains('342').should('be.visible');
    cy.contains('Resolution Rate').should('be.visible');
    cy.contains('91%').should('be.visible');
  });

  it('opens a recent interaction from the dashboard', () => {
    cy.get('a[href^="/manager/inspector/int-"]').first().click();

    cy.wait('@getInteractionDetail');
    cy.location('pathname').should('match', /\/manager\/inspector\/.+/);
    cy.contains('Back to Sessions').should('be.visible');
  });

  it('opens the full session inspector list', () => {
    cy.contains('View All Interactions').click();

    cy.wait('@getInteractions');
    cy.location('pathname').should('eq', '/manager/inspector');
    cy.contains('Session Inspector').should('be.visible');
  });
});
