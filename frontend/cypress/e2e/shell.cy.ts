describe('Shared shell behavior', () => {
  it('opens the profile screen from the user menu', () => {
    cy.loginAs('manager');
    cy.wait('@getDashboardStats');

    cy.get('[data-cy="user-menu-trigger"]').click();
    cy.contains('Profile').click();

    cy.location('pathname').should('eq', '/manager/settings');
    cy.location('hash').should('eq', '#profile');
    cy.contains('Profile Information').should('be.visible');
    cy.contains('Manager King').should('be.visible');
  });

  it('persists the selected theme after a reload', () => {
    cy.loginAs('manager');
    cy.wait('@getDashboardStats');

    cy.get('[data-cy="user-menu-trigger"]').click();
    cy.get('[data-cy="theme-option-dark"]').click();

    cy.document().its('documentElement').should('have.class', 'dark');

    cy.reload();
    cy.wait('@getUserMe');
    cy.wait('@getDashboardStats');

    cy.document().its('documentElement').should('have.class', 'dark');
    cy.window().then((win) => {
      expect(win.localStorage.getItem('vocalmind-theme')).to.equal('dark');
    });
  });

  it('uses the under-development back action to return to the previous screen', () => {
    cy.loginAs('agent');
    cy.wait('@getAgents');
    cy.wait('@getAgentProfile');

    cy.contains('My Calls').click();
    cy.contains('Under Development').should('be.visible');

    cy.get('[data-cy="under-development-back"]').click();

    cy.location('pathname').should('eq', '/agent');
    cy.contains('My Performance').should('be.visible');
  });

  it('uses the under-development home action to return to the manager dashboard', () => {
    cy.visitAs('manager', '/manager/coming-soon');

    cy.contains('Under Development').should('be.visible');
    cy.get('[data-cy="under-development-home"]').click();
    cy.wait('@getDashboardStats');

    cy.location('pathname').should('eq', '/manager');
    cy.contains('Average Score').should('be.visible');
  });
});
