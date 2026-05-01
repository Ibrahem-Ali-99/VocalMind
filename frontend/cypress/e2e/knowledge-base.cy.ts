describe("Knowledge Base", () => {
  beforeEach(() => {
    cy.visitAs("manager", "/manager/knowledge");
    cy.wait(["@getPolicies", "@getFaqs", "@getKB"]);
  });

  it("renders the info banner", () => {
    cy.contains("Knowledge Engine");
    cy.contains("Define the criteria and behavioral guardrails for your AI evaluation pipeline.");
  });

  it("displays the tabs correctly", () => {
    cy.contains("button", "Policies");
    cy.contains("button", "SOPs");
    cy.contains("button", "Knowledge Base");
  });

  it("displays the summary stats", () => {
    cy.contains("Active Policies");
    cy.contains("SOP Coverage");
    cy.contains("KB Articles");
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

  it("lists all SOP articles with questions and categories", () => {
    cy.contains("button", "SOPs").click();
    cy.contains("How do I reset a customer's password?");
    cy.contains("Account Management");
    cy.contains("What is the refund policy?");
    cy.contains("Billing");
  });

  it("lists all KB articles with titles and categories", () => {
    cy.contains("button", "Knowledge Base").click();
    cy.contains("Technical Specs V2");
    cy.contains("Hardware");
  });

  it("has toggle switches for items", () => {
    // Radix UI Switch renders with role="switch"
    cy.get('button[role="switch"]').should("have.length.at.least", 3);
  });

  it("filters policies by search", () => {
    cy.get('input[placeholder="Search policies..."]').type("Greeting");
    cy.contains("Greeting Protocol").should("be.visible");
    cy.contains("Data Privacy Guidelines").should("not.exist");
    cy.contains("Escalation Procedure").should("not.exist");
  });

  it("filters SOP articles by search", () => {
    cy.contains("button", "SOPs").click();
    cy.get('input[placeholder="Search SOPs..."]').type("refund");
    cy.contains("What is the refund policy?").should("be.visible");
    cy.contains("How do I reset a customer's password?").should("not.exist");
  });

  it("filters KB articles by search", () => {
    cy.contains("button", "Knowledge Base").click();
    cy.get('input[placeholder="Search knowledge base..."]').type("Specs");
    cy.contains("Technical Specs V2").should("be.visible");
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
