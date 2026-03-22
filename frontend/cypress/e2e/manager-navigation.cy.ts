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

    // Click collapse button (Menu icon)
    cy.get('button[title="Collapse sidebar"]').click();

    // After collapse, nav labels should be hidden
    cy.contains("span", "Dashboard").should("not.exist");

    // Click expand button
    cy.get('button[title="Expand sidebar"]').click();

    // Labels restored
    cy.contains("span", "Dashboard").should("be.visible");
  });
});
