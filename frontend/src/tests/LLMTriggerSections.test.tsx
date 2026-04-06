import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { SessionDetail } from "../app/components/manager/SessionDetail";
import { AgentCallDetail } from "../app/components/agent/AgentCallDetail";

vi.mock("../app/services/api", () => {
  const mockDetail = {
    interaction: {
      id: "int-100",
      agentName: "Agent A",
      agentId: "agent-1",
      date: "2026-03-21",
      time: "10:00 AM",
      duration: "3:00",
      language: "en",
      overallScore: 82,
      empathyScore: 80,
      policyScore: 78,
      resolutionScore: 75,
      resolved: false,
      hasViolation: true,
      hasOverlap: false,
      responseTime: "1.1s",
      status: "completed",
      audioFilePath: null,
    },
    utterances: [],
    emotionEvents: [],
    policyViolations: [],
    emotionComparison: {
      totalUtterances: 0,
      distributions: { acoustic: [], text: [], fused: [] },
      quality: {
        acousticTextAgreementRate: 0,
        fusedMatchesAcousticRate: 0,
        fusedMatchesTextRate: 0,
        disagreementCount: 0,
      },
    },
    llmTriggers: {
      available: true,
      emotionShift: {
        isDissonanceDetected: true,
        dissonanceType: "Sarcasm",
        rootCause: "insufficient evidence",
        counterfactualCorrection: "If the agent had acknowledged the customer concern first, escalation might have dropped.",
        evidenceQuotes: [],
        citations: [],
      },
      processAdherence: {
        detectedTopic: "billing_issue",
        isResolved: false,
        efficiencyScore: 6,
        justification: "Agent missed one verification step.",
        missingSopSteps: ["Confirm account details"],
        evidenceQuotes: [],
        citations: [],
      },
      nliPolicy: {
        nliCategory: "Contradiction",
        justification: "Agent statement conflicts with policy.",
        evidenceQuotes: [],
        citations: [],
      },
      explainability: {
        triggerAttributions: [
          {
            attributionId: "sop-1",
            family: "sop",
            triggerType: "SOP Violation",
            title: "Confirm account details",
            verdict: "Contradiction",
            confidence: 0.73,
            evidenceSpan: {
              utteranceIndex: 0,
              speaker: "agent",
              quote: "Let me look up the billing record.",
              timestamp: "00:15",
              startSeconds: 15,
              endSeconds: 18,
            },
            policyReference: {
              source: "sop",
              reference: "Billing SOP",
              clause: "Verify account and charge details before lookup.",
              provenance: "SOP retrieval context",
            },
            reasoning: "The agent moved to account lookup before verification.",
            evidenceChain: ["Expected SOP step: Confirm account details."],
            supportingQuotes: ["Let me look up the billing record."],
          },
        ],
        claimProvenance: [
          {
            claimId: "claim-1",
            claimText: "The refund will clear today.",
            claimSpan: {
              utteranceIndex: 0,
              speaker: "agent",
              quote: "The refund will clear today.",
              timestamp: "00:15",
              startSeconds: 15,
              endSeconds: 18,
            },
            retrievedPolicy: {
              source: "policy",
              reference: "Refund Policy",
              clause: "Standard refunds take 3-5 business days.",
              provenance: "Refund Timelines",
            },
            semanticSimilarity: 0.79,
            nliVerdict: "Contradiction",
            confidence: 0.81,
            reasoning: "The promise contradicts the retrieved policy clause.",
            provenance: "Refund Policy • Refund Timelines",
            supportingQuotes: ["The refund will clear today."],
          },
        ],
      },
    },
  };

  return {
    getInteractionDetail: vi.fn(async () => mockDetail),
    getAudioUrl: vi.fn(() => ""),
  };
});

describe("LLM trigger sections", () => {
  it("renders manager llm trigger section", async () => {
    render(
      <MemoryRouter initialEntries={["/manager/inspector/int-100"]}>
        <Routes>
          <Route path="/manager/inspector/:id" element={<SessionDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Emotion Trigger Reasoning")).toBeInTheDocument();
    expect(screen.getByText(/billing_issue/)).toBeInTheDocument();
    expect(screen.getAllByText(/Contradiction/).length).toBeGreaterThan(0);
    expect(screen.getByText("Evidence-Anchored Explainability")).toBeInTheDocument();
  });

  it("renders agent llm coaching section", async () => {
    render(
      <MemoryRouter initialEntries={["/agent/int-100"]}>
        <Routes>
          <Route path="/agent/:id" element={<AgentCallDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("LLM Coaching Insights")).toBeInTheDocument();
    expect(screen.getByText("billing_issue")).toBeInTheDocument();
    expect(screen.getAllByText("Contradiction").length).toBeGreaterThan(0);
  });
});
