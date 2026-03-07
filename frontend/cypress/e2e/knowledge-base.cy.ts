describe("Knowledge Base", () => {
  beforeEach(() => {
    cy.visit("/manager/knowledge");
  });

  it("renders the info banner", () => {
    cy.contains("Manage which policies and FAQ articles are active");
    cy.contains("Deactivating a policy removes it from future call evaluations");
  });

  it("displays the Company Policies section", () => {
    cy.contains("h3", "Company Policies");
    cy.contains("company_policies JOIN organization_policies");
  });

  it("displays the FAQ Articles section", () => {
    cy.contains("h3", "FAQ Articles");
    cy.contains("faq_articles JOIN organization_faq_articles");
  });

  it("lists all policies with titles and categories", () => {
    cy.contains("Greeting Protocol");
    cy.contains("Communication");
    cy.contains("Data Privacy Guidelines");
    cy.contains("Security");
    cy.contains("Escalation Procedure");
    cy.contains("Process");
  });

  it("lists all FAQ articles with questions and categories", () => {
    cy.contains("How do I reset a customer's password?");
    cy.contains("Account Management");
    cy.contains("What is the refund policy?");
    cy.contains("Billing");
  });

  it("shows active/inactive status labels for policies", () => {
    // Two active, one inactive in mock data
    cy.contains("Active");
    cy.contains("Inactive");
  });

  it("has toggle switches for policies", () => {
    // Radix UI Switch renders with role="switch"
    cy.get('button[role="switch"]').should("have.length.at.least", 3);
  });

  it("filters policies by search", () => {
    cy.get('input[placeholder="Search policies..."]').type("Greeting");
    cy.contains("Greeting Protocol").should("be.visible");
    cy.contains("Data Privacy Guidelines").should("not.exist");
    cy.contains("Escalation Procedure").should("not.exist");
  });

  it("filters FAQ articles by search", () => {
    cy.get('input[placeholder="Search FAQs..."]').type("refund");
    cy.contains("What is the refund policy?").should("be.visible");
    cy.contains("How do I reset a customer's password?").should("not.exist");
  });

  it("clears search to show all items again", () => {
    cy.get('input[placeholder="Search policies..."]').type("Greeting");
    cy.contains("Greeting Protocol").should("be.visible");
    cy.get('input[placeholder="Search policies..."]').clear();
    cy.contains("Greeting Protocol").should("be.visible");
    cy.contains("Data Privacy Guidelines").should("be.visible");
    cy.contains("Escalation Procedure").should("be.visible");
  });
});
