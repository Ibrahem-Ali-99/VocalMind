describe("Agent Dashboard", () => {
  beforeEach(() => {
    cy.visit("/agent");
  });

  it("renders agent performance stats", () => {
    cy.contains("Avg Score");
    cy.contains("88%");
    cy.contains("Resolution");
    cy.contains("91%");
  });

  it("renders performance charts", () => {
    cy.get(".recharts-wrapper").should("have.length.at.least", 1);
  });

  it("displays recent calls", () => {
    cy.contains("09:15 AM");
    cy.contains("Review");
  });

  it("renders the hero card with agent info", () => {
    cy.contains("MY PERFORMANCE");
    cy.contains("Sarah M.");
    cy.contains("Agent · VocalMind Corp");
  });

  it("shows score breakdown progress bars", () => {
    cy.contains("My Score Breakdown");
    cy.contains("Empathy Score");
    cy.contains("Policy Adherence");
  });

  it("shows weekly trend chart heading", () => {
    cy.contains("My Weekly Trend");
  });

  it("recent calls link to call detail", () => {
    cy.contains("a", "09:15 AM").click();
    cy.url().should("include", "/agent/calls/");
  });
});
