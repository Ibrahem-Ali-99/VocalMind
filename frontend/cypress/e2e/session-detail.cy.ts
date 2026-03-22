describe("Session Detail", () => {
  beforeEach(() => {
    // Navigate to the first interaction's detail page
    cy.visit("/manager/inspector/int-001");
  });

  it("renders the back navigation link", () => {
    cy.contains("a", "Back to Session Inspector")
      .should("have.attr", "href", "/manager/inspector");
  });

  it("displays agent name and call metadata", () => {
    cy.contains("h2", "Sarah M.");
    cy.contains("SESSION INSPECTOR");
    cy.contains("English");
  });

  it("displays the score grid with four categories", () => {
    cy.contains("Empathy");
    cy.contains("Policy");
    cy.contains("Resolution");
    cy.contains("Resp. Time");
  });

  it("renders the transcript section with utterances", () => {
    cy.contains("h3", "Transcript");
    cy.contains("utterances ordered by sequence_index");
    // Verify utterance text from mock data
    cy.contains("Good morning! Thank you for calling VocalMind support.");
    cy.contains("Hi, I've been having issues with my account login");
  });

  it("renders emotion events section", () => {
    cy.contains("h3", "Emotion Events");
    cy.contains("emotion_events — AI-detected emotional shifts");
    // Verify emotion transitions from mock data (case-insensitive due to CSS capitalize)
    cy.contains(/neutral/i);
    cy.contains(/frustrated/i);
  });

  it("shows Jump-to buttons for emotion events", () => {
    cy.contains("button", "Jump to 00:05");
  });

  it("supports RLHF feedback flow — flag as incorrect", () => {
    // Click the first "Flag as incorrect" button
    cy.contains("button", "Flag as incorrect").first().click();
    // Should now show the feedback question
    cy.contains("Was this detection accurate?");
    cy.contains("button", "Accurate");
    cy.contains("button", "Incorrect");
  });

  it("records feedback after thumbs up", () => {
    cy.contains("button", "Flag as incorrect").first().click();
    cy.contains("button", "Accurate").click();
    cy.contains("Feedback recorded — queued for model retraining");
  });

  it("renders policy violations section", () => {
    cy.contains("h3", "Policy Violations");
    cy.contains("policy_compliance WHERE is_compliant = FALSE");
    // Mock data has violations
    cy.contains("Hold Time Limit");
    cy.contains("Escalation Policy");
  });

  it("supports RLHF feedback on violations", () => {
    // Find a violation flag button and click it
    cy.get('[class*="bg-destructive/5"]').first().within(() => {
      cy.contains("button", "Flag as incorrect").click();
    });
    cy.contains("Was this verdict correct?");
    cy.contains("button", "Correct");
    cy.contains("button", "Incorrect");
  });

  it("navigates back to session inspector", () => {
    cy.contains("a", "Back to Session Inspector").click();
    cy.url().should("include", "/manager/inspector");
    cy.url().should("not.include", "/int-001");
  });
});
