describe("Landing Page", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  it("renders the application title and tagline", () => {
    cy.contains("h1", "VocalMind");
    cy.contains("AI-Powered Call Centre Evaluation Platform");
  });

  it("displays both portal cards", () => {
    cy.contains("h2", "Manager Portal");
    cy.contains("Full org access");
    cy.contains("h2", "Agent Portal");
    cy.contains("Personal view only");
  });

  it("navigates to the Manager Portal", () => {
    cy.contains("Enter Manager Portal →").click();
    cy.url().should("include", "/manager");
    cy.contains("Dashboard");
  });

  it("navigates to the Agent Portal", () => {
    cy.contains("Enter Agent Portal →").click();
    cy.url().should("include", "/agent");
    cy.contains("My Performance");
  });

  it("displays the footer note", () => {
    cy.contains("Each portal provides a tailored experience based on your role");
  });

  it("lists feature descriptions for each portal", () => {
    cy.contains("Comprehensive dashboard with org-wide KPIs");
    cy.contains("Session Inspector with emotion detection");
    cy.contains("Personal performance metrics and trends");
    cy.contains("Customer emotion journey insights");
  });
});
