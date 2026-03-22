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

  it("displays recent interactions with data", () => {
    cy.contains("Recent Interactions");
    cy.contains("Sarah M.");
    cy.contains("VIOLATION");
  });

  it("interaction cards link to session detail", () => {
    cy.get('a[href^="/manager/inspector/"]').first().click();
    cy.url().should("match", /\/manager\/inspector\/.+/);
  });

  it("renders chart section headings", () => {
    cy.contains("Weekly Score Trends");
    cy.contains("Emotion Distribution");
    cy.contains("Agent Leaderboard");
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
