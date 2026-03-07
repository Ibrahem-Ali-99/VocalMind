describe("Session Inspector", () => {
  beforeEach(() => {
    cy.visit("/manager/inspector");
  });

  it("renders the page heading and subtitle", () => {
    cy.contains("h2", "Session Inspector");
    cy.contains("All interactions · sorted by score");
  });

  it("displays search input and filter controls", () => {
    cy.get('input[placeholder="Search agent, date, ID…"]').should("exist");
    cy.contains("button", "All Agents");
    cy.contains("button", "Score ↑");
    cy.contains("button", "Date ↓");
    cy.contains("button", "Duration");
  });

  it("renders table headers", () => {
    cy.contains("Agent");
    cy.contains("Date & Time");
    cy.contains("Duration");
    cy.contains("Score");
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
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
    cy.contains("⚠ Violation");
  });

  it("has Inspect links that navigate to session detail", () => {
    cy.contains("a", "Inspect →").first().click();
    cy.url().should("match", /\/manager\/inspector\/.+/);
  });

  it("displays pagination footer", () => {
    cy.contains("Showing 1–4 of 342");
    cy.contains("button", "← Prev").should("be.disabled");
    cy.contains("button", "Next →").should("not.be.disabled");
  });
});
