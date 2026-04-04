import { buildInteractionSummary } from '../support/mockApi';

const inspectorInteractions = Array.from({ length: 12 }, (_, index) => {
  const number = index + 1;
  const paddedNumber = String(number).padStart(2, '0');
  const agentName =
    number % 3 === 0 ? 'John D.' : number % 2 === 0 ? 'Emily R.' : 'Sarah M.';

  return buildInteractionSummary({
    id: `int-2${paddedNumber}`,
    agentId: `agent-${String((number % 3) + 1).padStart(3, '0')}`,
    agentName,
    overallScore: 60 + number * 3,
    date: `2025-03-${paddedNumber}`,
    time: `${String(8 + (number % 4)).padStart(2, '0')}:00 AM`,
    duration: `${paddedNumber}:00`,
    resolved: number % 2 === 0,
    hasViolation: number % 4 === 0,
    responseTime: `${(1 + number / 10).toFixed(1)}s`,
  });
});

describe('Session Inspector', () => {
  beforeEach(() => {
    cy.visitAs('manager', '/manager/inspector', {
      interactions: {
        body: inspectorInteractions,
      },
    });
    cy.wait('@getInteractions');
  });

  it('filters the interaction list and opens the selected session', () => {
    cy.get('input[placeholder*="Search agent"]').type('John');

    cy.get('tbody').within(() => {
      cy.contains('John D.').should('be.visible');
      cy.contains('Sarah M.').should('not.exist');
    });

    cy.get('tbody').contains('John D.').parents('tr').within(() => {
      cy.contains('Inspect').click();
    });

    cy.wait('@getInteractionDetail');
    cy.location('pathname').should('match', /\/manager\/inspector\/.+/);
    cy.contains('Back to Sessions').should('be.visible');
  });

  it('filters the list by agent from the dropdown', () => {
    cy.get('select').select('Emily R.');

    cy.get('tbody').within(() => {
      cy.contains('Emily R.').should('be.visible');
      cy.contains('John D.').should('not.exist');
      cy.contains('Sarah M.').should('not.exist');
    });
  });

  it('reorders the table when duration sorting is toggled', () => {
    cy.contains('button', 'Duration').click();
    cy.contains('button', 'Duration').click();

    cy.get('tbody tr').first().should('contain', '01:00');
  });

  it('shows an empty state when no interactions match the filters', () => {
    cy.get('input[placeholder*="Search agent"]').type('No matching session');

    cy.contains('No interactions found').should('be.visible');
    cy.contains('Try adjusting your filters or search query.').should(
      'be.visible',
    );
  });

  it('paginates when more than ten interactions are available', () => {
    cy.get('tbody').should('not.contain', '01:00');
    cy.contains('button', 'Next').click();

    cy.get('tbody').should('contain', '02:00');
    cy.get('tbody').should('contain', '01:00');
  });
});
