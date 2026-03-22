describe("Session Detail", () => {
  beforeEach(() => {
    cy.visit("/manager/inspector/int-001");
  });

  it("renders the back navigation link", () => {
    cy.contains("a", "Back to Sessions")
      .should("have.attr", "href", "/manager/inspector");
  });

  it("displays agent name and call metadata", () => {
    cy.contains("h2", "Sarah M.");
    cy.contains(/english/i);
    cy.contains("ID:");
  });

  it("displays the score grid with four categories", () => {
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
    cy.contains("Response Time");
  });

  it("renders the transcript section with utterances", () => {
    cy.contains("h3", "Session Transcript");
    cy.contains("Good morning! Thank you for calling VocalMind support.");
    cy.contains("Hi, I've been having issues with my account login");
  });

  it("renders emotion graph section", () => {
    cy.contains("Emotion Graph");
    cy.contains("Agent");
    cy.contains("Customer");
    cy.contains("Playback");
  });

  it("renders automated evaluation cards", () => {
    cy.contains("h3", "Automated Evaluation");
    cy.contains("Process Adherence");
    cy.contains("Policy Inference");
  });

  it("renders emotion trigger reasoning card", () => {
    cy.contains("h3", "Emotion Trigger Reasoning");
    cy.contains("Current Customer Emotion");
    cy.contains("AI REASONING");
  });

  it("navigates back to session inspector", () => {
    cy.contains("a", "Back to Sessions").click();
    cy.url().should("include", "/manager/inspector");
    cy.url().should("not.include", "/int-001");
  });
});
