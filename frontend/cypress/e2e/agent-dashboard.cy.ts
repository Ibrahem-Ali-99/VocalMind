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
});
