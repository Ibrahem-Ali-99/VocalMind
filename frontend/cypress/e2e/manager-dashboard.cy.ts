describe("Manager Dashboard", () => {
  beforeEach(() => {
    cy.visit("/manager");
  });

  it("renders KPI stat cards", () => {
    cy.contains("Average Score");
    cy.contains("Calls Processed");
    cy.contains("Resolution Rate");
    cy.contains("Policy Violations");
  });

  it("renders chart containers", () => {
    cy.get(".recharts-wrapper").should("have.length.at.least", 2);
  });

  it("displays the interactions table with data", () => {
    cy.contains("Agent");
    cy.contains("Score");
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
    // Verify at least one row with mock data agent name
    cy.contains("Sarah M.");
  });

  it("interaction cards link to session detail", () => {
    cy.contains("a", "Sarah M.").first().click();
    cy.url().should("match", /\/manager\/inspector\/.+/);
  });
});
