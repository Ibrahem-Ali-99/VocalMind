describe("Manager Assistant", () => {
  beforeEach(() => {
    cy.visit("/manager/assistant");
  });

  it("renders the assistant header", () => {
    cy.contains("h2", "Manager Assistant");
    cy.contains("Ask anything about your call centre");
  });

  it("displays the initial assistant welcome message", () => {
    cy.contains("Voice or text — queries are logged to assistant_queries");
  });

  it("shows suggested query buttons", () => {
    cy.contains("Suggested queries");
    cy.contains("button", "Show top performing agents this week");
    cy.contains("button", "List all policy violations today");
    cy.contains("button", "Which agent has the lowest resolution rate?");
    cy.contains("button", "Show emotion trends across all calls");
  });

  it("sends a message when a suggested query is clicked", () => {
    cy.contains("button", "Show top performing agents this week").click();
    cy.contains("Show top performing agents this week");
    cy.contains("I've analyzed your query.");
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
    cy.contains("Executed in 120ms");
  });

  it("sends a message via Enter key", () => {
    cy.get('input[placeholder="Ask about scores, violations, agent trends…"]')
      .type("Show me violations{enter}");
    cy.contains("Show me violations");
    cy.contains("I've analyzed your query.");
  });

  it("does not send empty messages", () => {
    // Send button should be disabled when input is empty
    cy.get('button').filter(':has(svg.lucide-send)').should("be.disabled");
  });

  it("has a microphone button for voice input", () => {
    cy.get('svg.lucide-mic').should("have.length.at.least", 1);
  });

  it("renders the text input and send button", () => {
    cy.get('input[placeholder="Ask about scores, violations, agent trends…"]').should("exist");
    cy.get('svg.lucide-send').should("exist");
  });
});
