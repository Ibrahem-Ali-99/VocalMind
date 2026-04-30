describe('Authentication and routing', () => {
  it('redirects guests away from protected routes', () => {
    cy.mockApiScenario({ auth: false });

    cy.visit('/manager');

    cy.location('pathname').should('eq', '/login');
    cy.contains('Welcome to VocalMind').should('be.visible');
    cy.get('input[type="email"]').should('be.visible');
  });

  it('logs in as a manager from the login page', () => {
    cy.mockApiScenario({ auth: false });

    cy.visit('/login');
    cy.get('input[type="email"]').type('manager@niletech.com');
    cy.get('input[type="password"]').type('password');
    cy.contains('button', 'Sign In').click();

    cy.location('pathname').should('eq', '/manager');
    cy.contains('Dashboard').should('be.visible');
    cy.contains('Average Score').should('be.visible');
  });

  it('logs in as an agent from the login page', () => {
    cy.mockApiScenario({ auth: false });

    cy.visit('/login');
    cy.get('input[type="email"]').type('mohsen@niletech.com');
    cy.get('input[type="password"]').type('password');
    cy.contains('button', 'Sign In').click();

    cy.location('pathname').should('eq', '/agent');
    cy.contains('Agent Portal').should('be.visible');
    cy.contains('My Performance').should('be.visible');
  });

  it('logs out and returns to the login screen', () => {
    cy.loginAs('manager');
    cy.wait('@getDashboardStats');

    cy.get('[data-cy="user-menu-trigger"]').click();
    cy.contains('Log out').click();
    cy.get('[data-cy="logout-confirm"]').click();

    cy.wait('@logout');
    cy.location('pathname').should('eq', '/login');
    cy.contains('Sign in to your dashboard').should('be.visible');
  });
});
