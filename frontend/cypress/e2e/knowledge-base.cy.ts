describe('Knowledge Base', () => {
  beforeEach(() => {
    cy.visitAs('manager', '/manager/knowledge');
    cy.wait('@getPolicies');
    cy.wait('@getFaqs');
  });

  it('filters the policy list from the search field', () => {
    cy.get('input[placeholder="Search policies..."]').type('Greeting');

    cy.contains('Greeting Protocol').should('be.visible');
    cy.contains('Escalation Procedure').should('not.exist');
  });

  it('filters the faq list from the search field', () => {
    cy.get('input[placeholder="Search FAQs..."]').type('refund');

    cy.contains('What is the refund policy?').should('be.visible');
    cy.contains("How do I reset a customer's password?").should('not.exist');
  });

  it('toggles a policy from inactive to active', () => {
    cy.get('[data-cy="policy-toggle-pol-003"]')
      .siblings('span')
      .should('contain', 'INACTIVE');

    cy.get('[data-cy="policy-toggle-pol-003"]').click();

    cy.get('[data-cy="policy-toggle-pol-003"]')
      .siblings('span')
      .should('contain', 'ACTIVE');
  });

  it('toggles an faq from active to inactive', () => {
    cy.get('[data-cy="faq-toggle-faq-002"]')
      .siblings('span')
      .should('contain', 'ACTIVE');

    cy.get('[data-cy="faq-toggle-faq-002"]').click();

    cy.get('[data-cy="faq-toggle-faq-002"]')
      .siblings('span')
      .should('contain', 'INACTIVE');
  });
});
