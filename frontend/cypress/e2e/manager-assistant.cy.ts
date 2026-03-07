describe("Manager Assistant", () => {
  beforeEach(() => {
    cy.visit("/manager/assistant");
  });

  it("renders the assistant header", () => {
    cy.contains("h2", "Manager Assistant");
    cy.contains("Ask anything about your call centre");
  });

  it("displays the initial assistant welcome message", () => {
    cy.contains("Hello! I'm your VocalMind assistant.");
  });

  it("shows suggested query buttons", () => {
    cy.contains("Suggested queries");
    cy.contains("button", "Show top performing agents this week");
    cy.contains("button", "List all policy violations today");
    cy.contains("button", "Which agent has the lowest resolution rate?");
    cy.contains("button", "Show emotion trends across all calls");
  });

  it("fills input when a suggested query is clicked", () => {
    cy.contains("button", "Show top performing agents this week").click();
    cy.get('input[placeholder="Ask about scores, violations, agent trends…"]')
      .should("have.value", "Show top performing agents this week");
  });

  it("sends a message and receives an AI response", () => {
    cy.get('input[placeholder="Ask about scores, violations, agent trends…"]')
      .type("What is the average score?");
    // Click the send button
    cy.get('button').filter(':has(svg.lucide-send)').click();
    // User message should appear
    cy.contains("What is the average score?");
    // AI response should appear
    cy.contains("I've analyzed your query.");
    // SQL block should be visible
    cy.contains("SELECT * FROM interactions");
    // Execution time badge
    cy.contains("Executed in 98ms");
  });

  it("sends a message via Enter key", () => {
    cy.get('input[placeholder="Ask about scores, violations, agent trends…"]')
      .type("Show me violations{enter}");
    cy.contains("Show me violations");
    cy.contains("I've analyzed your query.");
  });

  it("does not send empty messages", () => {
    // Count existing messages
    cy.get('[class*="justify-end"], [class*="justify-start"]').then(($msgs) => {
      const initialCount = $msgs.length;
      // Try to send empty message
      cy.get('button').filter(':has(svg.lucide-send)').click();
      // Count should not increase
      cy.get('[class*="justify-end"], [class*="justify-start"]')
        .should("have.length", initialCount);
    });
  });

  it("has a microphone button for voice input", () => {
    cy.get('svg.lucide-mic').should("have.length.at.least", 1);
  });

  it("renders the text input and send button", () => {
    cy.get('input[placeholder="Ask about scores, violations, agent trends…"]').should("exist");
    cy.get('svg.lucide-send').should("exist");
  });
});
