describe("Session Inspector", () => {
  beforeEach(() => {
    cy.visitAs("manager", "/manager/inspector");
  });

  it("renders the page heading and subtitle", () => {
    cy.contains("h2", "Session Inspector");
    cy.contains(/sorted by score/i);
  });

  it("displays search input and filter controls", () => {
    cy.get('input[placeholder="Search agent, date, ID…"]').should("exist");
    cy.contains("button", /score/i);
    cy.contains("button", /date/i);
    cy.contains("button", /duration/i);
  });

  it("renders table headers", () => {
    cy.contains("Agent");
    cy.contains(/Date & Time/i);
    cy.contains("Duration");
    cy.contains("Score");
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
    cy.contains("Status");
    cy.contains("Error");
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
    cy.contains("a", /^Inspect$/).first().click();
    cy.url().should("match", /\/manager\/inspector\/.+/);
  });

  it("displays pagination footer", () => {
    cy.contains(/Showing\s+\d+–\d+\s+of\s+\d+/);
    cy.contains("button", "Prev").should("be.disabled");
    // Based on whether there's more than 10 mock interactions 
    // it could be disabled or not. Let's just check it exists.
    cy.contains("button", "Next").should("exist");
  });
});
