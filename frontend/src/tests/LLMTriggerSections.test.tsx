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
      processAdherence: {
        detectedTopic: "billing_issue",
        isResolved: false,
        efficiencyScore: 6,
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

    expect(await screen.findByText("LLM Trigger Status")).toBeInTheDocument();
    expect(screen.getByText("billing_issue")).toBeInTheDocument();
    expect(screen.getByText("Contradiction")).toBeInTheDocument();
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
    expect(screen.getByText("Contradiction")).toBeInTheDocument();
  });
});
