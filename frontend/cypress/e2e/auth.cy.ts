describe('Authentication', () => {
  it('should show login page', () => {
    cy.visit('/login')

    // Use more specific locator - target the main login heading
    cy.contains('h1, h2, h3, h4, h5, h6', 'VocalMind').should('be.visible')
    cy.get('input[type="email"]').should('be.visible')
    cy.get('input[type="password"]').should('be.visible')
    cy.contains('button', 'Sign In').should('be.visible')
  })

  it('should login with credentials', () => {
    cy.visit('/login')

    cy.get('input[type="email"]').type('galal@niletech.com')
    cy.get('input[type="password"]').type('password123')
    cy.contains('button', 'Sign In').click()

    // Should redirect to dashboard
    cy.url().should('match', /\/manager\/dashboard/)
  })

  it('should show SSO options', () => {
    cy.visit('/login')

    cy.contains('button', /Google/i).should('be.visible')
  })
})
