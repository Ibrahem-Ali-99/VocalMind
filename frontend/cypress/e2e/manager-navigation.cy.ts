describe("Manager Portal Navigation", () => {
  beforeEach(() => {
    cy.visit("/manager");
  });

  it("loads the manager layout with sidebar", () => {
    cy.contains("Manager Portal");
    cy.contains("Full org access");
    cy.contains("MK");
  });

  it("shows Dashboard as the default active page", () => {
    cy.get("h1").should("contain", "Dashboard");
  });

  it("navigates to Session Inspector", () => {
    cy.contains("Session Inspector").click();
    cy.url().should("include", "/manager/inspector");
    cy.get("h1").should("contain", "Session Inspector");
  });

  it("navigates to Manager Assistant", () => {
    cy.contains("Manager Assistant").click();
    cy.url().should("include", "/manager/assistant");
    cy.get("h1").should("contain", "Manager Assistant");
  });

  it("navigates to Knowledge Base", () => {
    cy.contains("Knowledge Base").click();
    cy.url().should("include", "/manager/knowledge");
    cy.get("h1").should("contain", "Knowledge Base");
  });

  it("collapses and expands the sidebar", () => {
    // Sidebar starts expanded — nav labels visible
    cy.contains("span", "Dashboard").should("be.visible");

    // Click collapse button (chevron-left)
    cy.get("button").find("svg.lucide-chevron-left").click();

    // After collapse, nav labels should be hidden
    cy.contains("span", "Dashboard").should("not.exist");

    // Click expand button (chevron-right)
    cy.get("button").find("svg.lucide-chevron-right").click();

    // Labels restored
    cy.contains("span", "Dashboard").should("be.visible");
  });
});
