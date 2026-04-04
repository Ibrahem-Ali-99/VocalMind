// Import custom Cypress commands
import './commands';

if (Cypress.env('coverage')) {
  require('@cypress/code-coverage/support');
}

beforeEach(() => {
  cy.clearCookies();
  cy.clearLocalStorage();
  cy.mockApiScenario();
});
