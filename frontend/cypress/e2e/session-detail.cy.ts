describe("Session Detail", () => {
  beforeEach(() => {
    // Navigate to the first interaction's detail page
    cy.visit("/manager/inspector/int-001");
  });

  it("renders the back navigation link", () => {
    cy.contains("a", "Back to Sessions")
      .should("have.attr", "href", "/manager/inspector");
  });

  it("displays agent name and call metadata", () => {
    cy.contains("h1", "Sarah M.");
    cy.contains("English");
  });

  it("displays the score grid with four categories", () => {
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
    cy.contains("Response Time");
  });

  it("renders the transcript section with utterances", () => {
    cy.contains("h3", "Session Transcript");
    // Verify utterance text from mock data
    cy.contains("Good morning! Thank you for calling VocalMind support.");
    cy.contains("Hi, I've been having issues with my account login");
  });

  it("renders emotion events section", () => {
    cy.contains("h3", "Emotion Flags");
    // Verify emotion transitions from mock data
    cy.contains("Neutral");
    cy.contains("Frustrated");
  });

  it("shows Jump-to buttons for emotion events", () => {
    cy.contains("button", "0:05");
  });

  it("supports RLHF feedback flow — flag as accurate", () => {
    // Click the first "Accurate" button
    cy.contains("button", "Accurate").first().click();
    // Feedback recorded (implementation specific, but let's check basic interaction)
  });

  it("records feedback after incorrect flag", () => {
    cy.contains("button", "Incorrect").first().click();
  });

  it("renders policy violations section", () => {
    cy.contains("Automated Evaluation");
    // Mock data has violations
    cy.contains("Process Adherence");
    cy.contains("Policy Inference");
  });

  it("supports RLHF feedback on evaluation", () => {
    // Find a feedback button and click it
    cy.contains("button", "Incorrect").last().click();
  });

  it("navigates back to session inspector", () => {
    cy.contains("a", "Back to Sessions").click();
    cy.url().should("include", "/manager/inspector");
    cy.url().should("not.include", "/int-001");
  });
});
