import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { SessionDetail } from '../app/components/manager/SessionDetail'
import { MemoryRouter, Routes, Route } from 'react-router'

const { getInteractionDetailMock } = vi.hoisted(() => ({
    getInteractionDetailMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getInteractionDetail: getInteractionDetailMock,
    getAudioUrl: vi.fn(() => ''),
    queryRag: vi.fn(async () => ({ response: 'ok', chunks: [], timing: {} })),
}))

const detail = {
    interaction: {
        id: 'int-001',
        agentName: 'Agent A',
        agentId: 'agent-1',
        date: '2026-03-21',
        time: '10:00 AM',
        duration: '3:00',
        language: 'en',
        overallScore: 82,
        empathyScore: 80,
        policyScore: 78,
        resolutionScore: 75,
        resolved: false,
        hasViolation: true,
        hasOverlap: false,
        responseTime: '1.1s',
        status: 'completed',
        audioFilePath: null,
    },
    utterances: [
        {
            id: 'u1',
            interactionId: 'int-001',
            speaker: 'agent',
            text: 'Hello, I can help with billing.',
            startTime: 0,
            endTime: 3,
            timestamp: '00:00',
            emotion: 'neutral',
            confidence: 0.8,
        },
    ],
    emotionEvents: [],
    policyViolations: [],
    emotionComparison: {
        totalUtterances: 1,
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
            detectedTopic: 'billing_issue',
            isResolved: false,
            efficiencyScore: 6,
            justification: 'Agent skipped one verification step.',
            missingSopSteps: ['Confirm account details'],
            evidenceQuotes: [],
            citations: [],
        },
        nliPolicy: {
            nliCategory: 'Contradiction',
            justification: 'Agent statement conflicts with policy.',
            evidenceQuotes: [],
            citations: [],
        },
    },
}

const renderWithId = (id = 'int-001') =>
    render(
        <MemoryRouter initialEntries={[`/manager/inspector/${id}`]}>
            <Routes>
                <Route path="/manager/inspector/:id" element={<SessionDetail />} />
            </Routes>
        </MemoryRouter>
    )

describe('SessionDetail', () => {
    it('renders session transcript and navigation', async () => {
        getInteractionDetailMock.mockResolvedValue(detail)
        renderWithId()

        expect(await screen.findByText('Back to Sessions')).toBeInTheDocument()
        expect(screen.getByText('Session Transcript')).toBeInTheDocument()
    })

    it('renders automated evaluation section with process and policy cards', async () => {
        getInteractionDetailMock.mockResolvedValue(detail)
        renderWithId()

        expect(await screen.findByText('Automated Evaluation')).toBeInTheDocument()
        expect(screen.getByText('Process Adherence')).toBeInTheDocument()
        expect(screen.getByText('Policy Inference')).toBeInTheDocument()
        expect(screen.getByText('billing_issue')).toBeInTheDocument()
        expect(screen.getByText('Contradiction')).toBeInTheDocument()
    })
})
