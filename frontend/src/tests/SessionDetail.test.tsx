import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
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
        {
            id: 'u2',
            interactionId: 'int-001',
            speaker: 'customer',
            text: 'This has been going on for days and I am frustrated.',
            startTime: 4,
            endTime: 8,
            timestamp: '00:04',
            emotion: 'fearful',
            confidence: 0.74,
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
        emotionShift: {
            isDissonanceDetected: true,
            dissonanceType: 'Sarcasm',
            rootCause: 'insufficient evidence',
            counterfactualCorrection: 'If the agent had acknowledged frustration first, escalation might have reduced.',
            evidenceQuotes: [],
            citations: [],
        },
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
        explainability: {
            triggerAttributions: [
                {
                    attributionId: 'emotion-1',
                    family: 'emotion',
                    triggerType: 'Acoustic-Transcript Dissonance',
                    title: 'Sarcasm',
                    verdict: 'Cross-Modal Mismatch',
                    confidence: 0.82,
                    evidenceSpan: {
                        utteranceIndex: 1,
                        speaker: 'customer',
                        quote: 'This has been going on for days and I am frustrated.',
                        timestamp: '00:04',
                        startSeconds: 4,
                        endSeconds: 8,
                    },
                    reasoning: 'Customer wording and fused emotion disagree.',
                    evidenceChain: [
                        'Acoustic emotion signal resolved to frustrated.',
                        'Transcript span used for review: This has been going on for days and I am frustrated.',
                    ],
                    supportingQuotes: ['This has been going on for days and I am frustrated.'],
                },
                {
                    attributionId: 'sop-1',
                    family: 'sop',
                    triggerType: 'SOP Violation',
                    title: 'Confirm account details',
                    verdict: 'Contradiction',
                    confidence: 0.74,
                    evidenceSpan: {
                        utteranceIndex: 0,
                        speaker: 'agent',
                        quote: 'Hello, I can help with billing.',
                        timestamp: '00:00',
                        startSeconds: 0,
                        endSeconds: 3,
                    },
                    policyReference: {
                        source: 'sop',
                        reference: 'billing-issue SOP',
                        clause: 'Verify account and charge details before lookup.',
                        provenance: 'SOP retrieval context',
                    },
                    reasoning: 'Agent skipped verification before continuing.',
                    evidenceChain: ['Expected SOP step: Confirm account details.'],
                    supportingQuotes: ['Hello, I can help with billing.'],
                },
            ],
            claimProvenance: [
                {
                    claimId: 'claim-1',
                    claimText: 'Your refund will arrive within 24 hours.',
                    claimSpan: {
                        utteranceIndex: 0,
                        speaker: 'agent',
                        quote: 'Your refund will arrive within 24 hours.',
                        timestamp: '00:00',
                        startSeconds: 0,
                        endSeconds: 3,
                    },
                    retrievedPolicy: {
                        source: 'policy',
                        reference: 'Refunds Policy v2.3',
                        clause: 'Standard refunds take 3-5 business days.',
                        provenance: 'Refund Timelines • v2.3',
                    },
                    semanticSimilarity: 0.81,
                    nliVerdict: 'Contradiction',
                    confidence: 0.8,
                    reasoning: 'The promise contradicts the active refund timeline.',
                    provenance: 'Refunds Policy v2.3 • Refund Timelines • v2.3',
                    supportingQuotes: ['Your refund will arrive within 24 hours.'],
                },
            ],
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

        expect(await screen.findByText('Back to Session Inspector')).toBeInTheDocument()
        expect(screen.getByText('Transcript')).toBeInTheDocument()
    })

    it('renders automated evaluation section with process and policy cards', async () => {
        getInteractionDetailMock.mockResolvedValue(detail)
        renderWithId()

        expect(await screen.findByText('Emotion Trigger Reasoning')).toBeInTheDocument()
        expect(screen.getByText('Automated Evaluation')).toBeInTheDocument()
        expect(screen.getByText(/Process Adherence/)).toBeInTheDocument()
        expect(screen.getByText(/Policy Inference/)).toBeInTheDocument()
        expect(screen.getByText(/billing_issue/)).toBeInTheDocument()
        expect(screen.getAllByText(/Contradiction/).length).toBeGreaterThan(0)
    })

    it('renders evidence-anchored explainability cards', async () => {
        getInteractionDetailMock.mockResolvedValue(detail)
        renderWithId()

        expect(await screen.findByText('Evidence-Anchored Explainability')).toBeInTheDocument()
        expect(screen.getByText('Claim to evidence to verdict')).toBeInTheDocument()

        fireEvent.click(screen.getByRole('button', { name: /Retrieval Provenance Scoring/i }))
        expect(screen.getByText('Retrieval Provenance Scoring')).toBeInTheDocument()
        expect(screen.getByText('Your refund will arrive within 24 hours.')).toBeInTheDocument()
        expect(screen.getByText('Standard refunds take 3-5 business days.')).toBeInTheDocument()
    })

    it('normalizes non-canonical emotions instead of treating them as neutral', async () => {
        getInteractionDetailMock.mockResolvedValue(detail)
        renderWithId()

        expect((await screen.findAllByText('Frustrated', { exact: false })).length).toBeGreaterThan(0)
    })

    it('shows confidence labels based on real utterance confidence values', async () => {
        getInteractionDetailMock.mockResolvedValue(detail)
        renderWithId()

        const confidenceElements = await screen.findAllByText('80', { exact: false })
        expect(confidenceElements.length).toBeGreaterThan(0)
        expect(screen.getByText('74', { exact: false })).toBeInTheDocument()
    })
})
