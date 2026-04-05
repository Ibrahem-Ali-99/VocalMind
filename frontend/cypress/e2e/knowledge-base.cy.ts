describe("Knowledge Base", () => {
  beforeEach(() => {
    cy.visit("/manager/knowledge");
    cy.wait(["@getPolicies", "@getFaqs"]);
  });

  it("renders the info banner", () => {
    cy.contains("Knowledge Engine");
    cy.contains("Define the criteria and behavioral guardrails for your AI evaluation pipeline.");
  });

  it("displays the Company Policies section", () => {
    cy.contains("button", "Guidelines");
    cy.contains("button", "SOP & Knowledge");
  });

  it("displays the FAQ Articles section", () => {
    cy.contains("SOP Coverage");
    cy.contains("Evaluation Hits");
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
    cy.contains("button", "SOP & Knowledge").click();
    cy.contains("How do I reset a customer's password?");
    cy.contains("Account Management");
    cy.contains("What is the refund policy?");
    cy.contains("Billing");
  });

  it("shows active/inactive status labels for policies", () => {
    cy.contains("Active Policies");
    cy.contains("SOP Coverage");
    cy.contains("Evaluation Hits");
  });

  it("has toggle switches for policies", () => {
    // Radix UI Switch renders with role="switch"
    cy.get('button[role="switch"]').should("have.length.at.least", 3);
  });

  it("filters policies by search", () => {
    cy.get('input[placeholder="Search guidelines..."]').type("Greeting");
    cy.contains("Greeting Protocol").should("be.visible");
    cy.contains("Data Privacy Guidelines").should("not.exist");
    cy.contains("Escalation Procedure").should("not.exist");
  });

  it("filters FAQ articles by search", () => {
    cy.contains("button", "SOP & Knowledge").click();
    cy.get('input[placeholder="Search SOP & knowledge..."]').type("refund");
    cy.contains("What is the refund policy?").should("be.visible");
    cy.contains("How do I reset a customer's password?").should("not.exist");
  });

  it("clears search to show all items again", () => {
    cy.get('input[placeholder="Search guidelines..."]').type("Greeting");
    cy.contains("Greeting Protocol").should("be.visible");
    cy.get('input[placeholder="Search guidelines..."]').clear();
    cy.contains("Greeting Protocol").should("be.visible");
    cy.contains("Data Privacy Guidelines").should("be.visible");
    cy.contains("Escalation Procedure").should("be.visible");
  });
});
