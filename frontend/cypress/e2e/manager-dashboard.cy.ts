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
    cy.get('a[href^="/manager/inspector/"]').first().click();
    cy.url().should("match", /\/manager\/inspector\/.+/);
  });

  it("renders chart section headings", () => {
    cy.contains("Weekly Score Trends");
    cy.contains("Emotion Distribution");
    cy.contains("Policy Compliance by Category");
    cy.contains("Agent Performance Breakdown");
  });

  it("displays the agent leaderboard", () => {
    cy.contains("Agent Leaderboard");
    cy.contains("Sarah M.");
    cy.contains("John D.");
  });

  it("shows recent interactions section", () => {
    cy.contains("Recent Interactions");
  });
});
