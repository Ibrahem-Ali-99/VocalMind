describe('Dashboard', () => {
  beforeEach(() => {
    // Login first
    cy.visit('/login')
    cy.get('input[type="email"]').type('galal@niletech.com')
    cy.get('input[type="password"]').type('password123')
    cy.contains('button', 'Sign In').click()
    cy.url().should('match', /dashboard/)
  })

  it('should display stats cards', () => {
    cy.contains('Welcome back').should('be.visible')
    cy.contains('Total Calls Today').should('be.visible')
    cy.contains('Avg Sentiment Score').should('be.visible')
  })

  it('should show flagged calls table', () => {
    cy.contains('Flagged Calls').should('be.visible')
    cy.contains('PRIORITY').should('be.visible')
  })

  it('should navigate to calls page', () => {
    cy.contains('View All').click()
    cy.url().should('match', /\/manager\/calls/)
  })
})
