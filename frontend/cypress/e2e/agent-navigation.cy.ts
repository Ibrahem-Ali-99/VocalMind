describe("Agent Portal Navigation", () => {
  beforeEach(() => {
    cy.visit("/agent");
  });

  it("loads the agent layout with sidebar", () => {
    cy.contains("Agent Portal");
    cy.contains("Personal view only");
    cy.contains("RK");
  });

  it("shows My Performance as the default page", () => {
    cy.get("h1").should("contain", "My Performance");
  });

  it("collapses and expands the sidebar", () => {
    cy.contains("span", "My Performance").should("be.visible");

    cy.get("button").find("svg.lucide-chevron-left").click();
    cy.contains("span", "My Performance").should("not.exist");

    cy.get("button").find("svg.lucide-chevron-right").click();
    cy.contains("span", "My Performance").should("be.visible");
  });
});
