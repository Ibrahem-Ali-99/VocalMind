import {
  registerApiScenario,
  type AppScenario,
  type TestRole,
} from './mockApi';

Cypress.Commands.add('mockApiScenario', (scenario: AppScenario = {}) => {
  registerApiScenario(scenario);
});

Cypress.Commands.add(
  'visitAs',
  (role: TestRole, path: string, scenario: AppScenario = {}) => {
    cy.clearCookies();
    cy.clearLocalStorage();
    cy.mockApiScenario({
      ...scenario,
      auth: { role },
    });
    cy.visit(path, {
      onBeforeLoad(win) {
        win.localStorage.setItem('vm_auth_cookie_hint', '1');
      },
    });
  },
);

Cypress.Commands.add(
  'loginAs',
  (role: TestRole, scenario: AppScenario = {}) => {
    const path = role === 'manager' ? '/manager' : '/agent';
    cy.visitAs(role, path, scenario);
  },
);

declare global {
  namespace Cypress {
    interface Chainable {
      loginAs(role: TestRole, scenario?: AppScenario): Chainable<void>;
      mockApiScenario(scenario?: AppScenario): Chainable<void>;
      visitAs(
        role: TestRole,
        path: string,
        scenario?: AppScenario,
      ): Chainable<void>;
    }
  }
}

export {};
