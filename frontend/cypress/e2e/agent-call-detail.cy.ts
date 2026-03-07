describe("Agent Call Detail", () => {
  beforeEach(() => {
    cy.visit("/agent/calls/int-001");
  });

  it("renders the back navigation link", () => {
    cy.contains("a", "Back to My Calls")
      .should("have.attr", "href", "/agent");
  });

  it("displays call header with date and time", () => {
    cy.contains("CALL DETAIL");
    cy.contains("27 Feb 2025 · 09:14");
    cy.contains("5:42 · ar-EG");
  });

  it("displays overall score ring", () => {
    cy.get("svg circle").should("have.length.at.least", 2);
    cy.contains("88%");
  });

  it("renders the score grid with four categories", () => {
    cy.contains("Empathy");
    cy.contains("85%");
    cy.contains("Policy");
    cy.contains("91%");
    cy.contains("Resolution");
    cy.contains("88%");
    cy.contains("Resp. Time");
    cy.contains("2.1s");
  });

  it("renders coaching points card with violations", () => {
    cy.contains("Coaching Points");
    cy.contains("Areas to focus on");
    cy.contains("Hold Time Limit");
    cy.contains("Escalation Policy");
  });

  it("displays violation scores", () => {
    cy.contains("Score: 45% — target 80%+");
    cy.contains("Score: 30% — target 80%+");
  });

  it("renders transcript section with utterances", () => {
    cy.contains("h3", "Transcript");
    cy.contains("Good morning! Thank you for calling VocalMind support.");
    cy.contains("Hi, I've been having issues with my account login");
    cy.contains("I'm sorry to hear that.");
  });

  it("shows speaker labels and emotion badges in transcript", () => {
    cy.contains("Me");
    cy.contains("Customer");
    cy.contains("Neutral");
    cy.contains("Frustrated");
  });

  it("renders customer emotion journey section", () => {
    cy.contains("h3", "Customer Emotion Journey");
    cy.contains("emotion_events — how customer sentiment changed");
  });

  it("shows emotion transitions with Jump buttons", () => {
    cy.contains("button", "Jump to 00:05");
  });

  it("navigates back to agent dashboard", () => {
    cy.contains("a", "Back to My Calls").click();
    cy.url().should("eq", Cypress.config().baseUrl + "/agent");
  });
});
