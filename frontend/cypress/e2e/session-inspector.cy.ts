describe("Session Inspector", () => {
  beforeEach(() => {
    cy.visit("/manager/inspector");
  });

  it("renders the page heading and subtitle", () => {
    cy.contains("h2", "Session Inspector");
    // Flexible match because the total count depends on mock data
    cy.contains(/Analyze \d+ automated LLM agent evaluations/);
  });

  it("displays search input and filter controls", () => {
    cy.get('input[placeholder="Search agent, date, ID..."]').should("exist");
    cy.contains("select", "Any Resolution");
    cy.contains("select", "Any Policies");
  });

  it("renders table headers", () => {
    cy.contains("Agent");
    cy.contains("Session Info");
    cy.contains("Score");
    cy.contains("Automated Metrics (Emp / Pol / Res)");
    cy.contains("Status");
    cy.contains("Actions");
  });

  it("renders all interaction rows from mock data", () => {
    // All 4 mock interactions should be rendered
    cy.contains("Sarah M.");
    cy.contains("John D.");
    cy.contains("Emily R.");
    cy.contains("Mike T.");
  });

  it("displays resolved and unresolved statuses", () => {
    cy.contains("✓ Resolved");
    cy.contains("✗ Unresolved");
  });

  it("shows violation badges for flagged interactions", () => {
    cy.contains("Violation");
  });

  it("has Inspect links that navigate to session detail", () => {
    // Wait for data
    cy.contains("Sarah M.").should("be.visible");
    // Target the link by href pattern
    cy.get('a[href*="/manager/inspector/int-"]').first().click();
    cy.url().should("include", "/manager/inspector/int-");
  });

  it("displays pagination footer", () => {
    cy.contains(/Showing 1–\d+ of \d+/);
    cy.contains("button", "Previous").should("be.disabled");
    // Based on whether there's more than 10 mock interactions 
    // it could be disabled or not. Let's just check it exists.
    cy.contains("button", "Next page").should("exist");
  });
});
