// Import custom Cypress commands
import './commands';

// Import code coverage support hooks
import '@cypress/code-coverage/support';

beforeEach(() => {
  cy.clearCookies();
  cy.clearLocalStorage();
});
