describe("Landing Page", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  it("links the manager portal card to the manager app", () => {
    cy.contains("Manager Portal").closest("a").should("have.attr", "href", "/manager");
  });

  it("links the agent portal card to the agent app", () => {
    cy.contains("Agent Portal").closest("a").should("have.attr", "href", "/agent");
  });
});
