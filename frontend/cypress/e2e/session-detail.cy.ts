describe("Session Detail", () => {
  beforeEach(() => {
    cy.visitAs("manager", "/manager/inspector/int-001");
  });

  it("renders the back navigation link", () => {
    cy.contains("a", "Back to Session Inspector")
      .should("have.attr", "href", "/manager/inspector");
  });

  it("displays agent name and call metadata", () => {
    cy.contains("h2", "Sarah M.");
    cy.contains(/english/i);
    cy.contains("2025-03-01");
    cy.contains("09:15 AM");
  });

  it("displays the score grid with four categories", () => {
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
    cy.contains("Resp. Time");
  });

  it("renders the transcript section with utterances", () => {
    cy.contains("h3", "Transcript");
    cy.contains("Good morning! Thank you for calling VocalMind support.");
    cy.contains("Hi, I've been having issues with my account login");
  });

  it("renders emotion events section", () => {
    cy.contains("Emotion Events");
    cy.contains("Agent");
    cy.contains("Customer");
    cy.contains("Jump to");
  });

  it("renders automated evaluation cards", () => {
    cy.contains("h3", "Automated Evaluation");
    cy.contains("Process Adherence");
    cy.contains("Policy Inference");
  });

  it("renders emotion trigger reasoning card", () => {
    cy.contains("h4", "Emotion Trigger Reasoning");
    cy.contains("Dissonance:");
    cy.contains("Counterfactual:");
  });

  it("navigates back to session inspector", () => {
    cy.contains("a", "Back to Session Inspector").click();
    cy.url().should("include", "/manager/inspector");
    cy.url().should("not.include", "/int-001");
  });
});
