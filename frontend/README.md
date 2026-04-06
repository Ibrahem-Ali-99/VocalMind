# VocalMind Frontend

Manager and agent web UI for VocalMind, built with React 18, Vite, Tailwind v4, MUI, and Radix UI.

## Main Areas

1. Manager dashboard and session inspector
2. Call detail pages with transcript playback and evaluation panels
3. Knowledge-base management
4. Assistant and agent-facing workflows

## Explainability UI

The manager session-detail page now includes the Evidence-Anchored Explainability panel.

It renders:

1. Span-Level Trigger Attribution cards
2. Retrieval Provenance Scoring cards
3. Timestamp jump actions that sync cards back to audio playback

See `docs/explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md` for the full feature contract.

## Development

Install dependencies:

```bash
npm install
```

Start the dev server:

```bash
npm run dev
```

Type-check:

```bash
npm run lint
```

Run tests:

```bash
npm run test
```

## Relevant Files

1. `src/app/components/manager/SessionInspector.tsx`
2. `src/app/components/manager/SessionDetail.tsx`
3. `src/app/components/manager/EvidenceAnchoredExplainabilityPanel.tsx`
4. `src/app/services/api.ts`

## Targeted UI Tests

1. `src/tests/SessionInspector.test.tsx`
2. `src/tests/SessionDetail.test.tsx`
3. `src/tests/LLMTriggerSections.test.tsx`
