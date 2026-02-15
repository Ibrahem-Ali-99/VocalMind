describe('Navigation', () => {
  beforeEach(() => {
    cy.visit('/login')
    cy.get('input[type="email"]').type('galal@niletech.com')
    cy.get('input[type="password"]').type('password123')
    cy.contains('button', 'Sign In').click()
    cy.url().should('match', /dashboard/)
  })

  it('should navigate to calls page', () => {
    cy.contains('Calls').click()
    cy.url().should('match', /\/manager\/calls/)
    cy.contains('All Calls').should('be.visible')
  })

  it('should navigate to team page', () => {
    cy.contains('Team').click()
    cy.url().should('match', /\/manager\/team/)
  })

  it('should navigate to settings page', () => {
    cy.contains('Settings').click()
    cy.url().should('match', /\/manager\/settings/)
    cy.contains('Profile Settings').should('be.visible')
  })

  it('should navigate to session inspector', () => {
    cy.visit('/session/CALL-2847')
    // Wait for text to be visible with timeout
    cy.contains('Call #CALL-2847', { timeout: 10000 }).should('be.visible')
  })
})
