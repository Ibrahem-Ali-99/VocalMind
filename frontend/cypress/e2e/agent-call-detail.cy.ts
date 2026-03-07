describe("Agent Call Detail", () => {
  describe("int-001 (High Score, No Violations)", () => {
    beforeEach(() => {
      cy.visit("/agent/calls/int-001");
    });

    it("renders the back navigation link", () => {
      cy.contains("a", "Back to My Calls").should("have.attr", "href", "/agent");
    });

    it("displays call header with updated date, time, and language", () => {
      cy.contains("CALL DETAIL");
      cy.contains("2025-03-01 · 09:15 AM");
      cy.contains("8:42 · English");
    });

    it("displays overall score ring", () => {
      cy.get("svg circle").should("have.length.at.least", 2);
      cy.contains("92%"); // Overall is 92 in mockData
    });

    it("renders the score grid with four categories", () => {
      cy.contains("Empathy");
      cy.contains("95%"); // Empathy for int-001
      cy.contains("Policy");
      cy.contains("90%");
      cy.contains("Resolution");
      cy.contains("88%");
      cy.contains("Resp. Time");
      cy.contains("1.2s");
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

  describe("int-002 (Low Score, Has Violations)", () => {
    beforeEach(() => {
      cy.visit("/agent/calls/int-002");
    });

    it("renders coaching points card with violations", () => {
      cy.contains("Coaching Points");
      cy.contains("Areas to focus on");
      cy.contains("Hold Time Limit");
    });

    it("displays violation scores", () => {
      cy.contains("Score: 45% — target 80%+");
    });
  });
});
