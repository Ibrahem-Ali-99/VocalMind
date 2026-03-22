import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { AgentCallDetail } from '../app/components/agent/AgentCallDetail'
import { MemoryRouter, Routes, Route } from 'react-router'

const { getInteractionDetailMock } = vi.hoisted(() => ({
    getInteractionDetailMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getInteractionDetail: getInteractionDetailMock,
    getAudioUrl: vi.fn(() => ''),
}))

const makeDetail = (id: string, overall = 85, violationTitle?: string) => ({
    interaction: {
        id,
        agentName: 'Agent A',
        agentId: 'agent-1',
        date: '2025-03-01',
        time: '09:14',
        duration: '4:20',
        language: 'en',
        overallScore: overall,
        empathyScore: overall,
        policyScore: overall,
        resolutionScore: overall,
        resolved: true,
        hasViolation: Boolean(violationTitle),
        hasOverlap: false,
        responseTime: '1.2s',
        status: 'completed',
        audioFilePath: null,
    },
    utterances: [
        {
            id: 'u1',
            interactionId: id,
            speaker: 'agent',
            text: 'Good morning!',
            startTime: 0,
            endTime: 2,
            timestamp: '00:00',
            emotion: 'happy',
            confidence: 0.9,
        },
    ],
    emotionEvents: [],
    policyViolations: violationTitle
        ? [
              {
                  id: 'v1',
                  interactionId: id,
                  policyName: 'hold_time_limit',
                  policyTitle: violationTitle,
                  category: 'operations',
                  description: 'desc',
                  reasoning: 'reason',
                  severity: 'medium',
                  score: 45,
              },
          ]
        : [],
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
    llmTriggers: null,
})

const renderWithId = (id: string) =>
    render(
        <MemoryRouter initialEntries={[`/agent/${id}`]}>
            <Routes>
                <Route path="/agent/:id" element={<AgentCallDetail />} />
            </Routes>
        </MemoryRouter>
    )

describe('AgentCallDetail', () => {
    it('renders call header and transcript', async () => {
        getInteractionDetailMock.mockResolvedValue(makeDetail('int-001', 85))
        renderWithId('int-001')

        expect(await screen.findByText('CALL DETAIL')).toBeInTheDocument()
        expect(screen.getByText('Transcript')).toBeInTheDocument()
        expect(screen.getByText(/Good morning!/)).toBeInTheDocument()
    })

    it('renders coaching points when policy violations exist', async () => {
        getInteractionDetailMock.mockResolvedValue(makeDetail('int-002', 85, 'Hold Time Limit'))
        renderWithId('int-002')

        expect(await screen.findByText('Coaching Points')).toBeInTheDocument()
        expect(screen.getByText('Hold Time Limit')).toBeInTheDocument()
    })

    it('renders back navigation link', async () => {
        getInteractionDetailMock.mockResolvedValue(makeDetail('int-001', 85))
        renderWithId('int-001')

        const link = await screen.findByText('Back to My Calls')
        expect(link.closest('a')).toHaveAttribute('href', '/agent')
    })

    it('renders mid-range score values', async () => {
        getInteractionDetailMock.mockResolvedValue(makeDetail('int-005', 78))
        renderWithId('int-005')

        expect(await screen.findByText(/2025-03-01/)).toBeInTheDocument()
        expect(screen.getByText('Empathy')).toBeInTheDocument()
    })
})
