describe('Agent Dashboard', () => {
  beforeEach(() => {
    cy.visit('/login')
    cy.get('input[type="email"]').type('galal@niletech.com')
    cy.get('input[type="password"]').type('password123')
    cy.contains('button', 'Sign In').click()
    cy.url().should('include', '/manager/dashboard')
  })

  it('should show agent dashboard', () => {
    cy.visit('/manager/dashboard')
    cy.contains('Welcome back').should('be.visible')
    cy.contains('Total Calls Today').should('be.visible')
  })

  it('should show manager dashboard', () => {
    cy.visit('/manager/dashboard')
    cy.contains('Flagged Calls').should('be.visible')
  })

  it('should navigate to team page', () => {
    cy.visit('/manager/team')
    cy.contains('Active Agents').should('be.visible')
  })

  it('should navigate to policies page', () => {
    cy.visit('/manager/policies')
    cy.contains('Company Policies').should('be.visible')
  })
})
