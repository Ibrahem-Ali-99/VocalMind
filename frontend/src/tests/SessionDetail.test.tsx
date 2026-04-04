import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SessionDetail } from '../app/components/manager/SessionDetail'
import { MemoryRouter, Routes, Route } from 'react-router'

const { getInteractionDetailMock, getAudioUrlMock } = vi.hoisted(() => ({
    getInteractionDetailMock: vi.fn(),
    getAudioUrlMock: vi.fn(),
}))

const mockPlay = vi.fn(() => Promise.resolve())
const mockPause = vi.fn()

vi.mock('../app/services/api', () => ({
    getInteractionDetail: getInteractionDetailMock,
    getAudioUrl: getAudioUrlMock,
}))

const baseDetail = {
    interaction: {
        id: 'int-001',
        agentName: 'Agent A',
        agentId: 'agent-1',
        date: '2026-03-21',
        time: '10:00 AM',
        duration: '0:12',
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
            text: 'Customer is getting upset about the hold time.',
            startTime: 5,
            endTime: 8,
            timestamp: '00:05',
            emotion: 'frustrated',
            confidence: 0.91,
        },
        {
            id: 'u3',
            interactionId: 'int-001',
            speaker: 'agent',
            text: 'I understand, let me fix that right away.',
            startTime: 9,
            endTime: 12,
            timestamp: '00:09',
            emotion: 'empathetic',
            confidence: 0.89,
        },
    ],
    emotionEvents: [],
    policyViolations: [],
    emotionComparison: {
        totalUtterances: 3,
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
            currentCustomerEmotion: 'frustrated',
            currentEmotionReasoning: 'The customer escalates after repeated holds.',
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
    },
}

type DetailOverrides = Partial<typeof baseDetail> & {
    ragCompliance?: any
    emotionTriggers?: any
}

function buildDetail(overrides: DetailOverrides = {}) {
    return {
        interaction: {
            ...baseDetail.interaction,
            ...overrides.interaction,
        },
        utterances: overrides.utterances ?? baseDetail.utterances,
        emotionEvents: overrides.emotionEvents ?? baseDetail.emotionEvents,
        policyViolations: overrides.policyViolations ?? baseDetail.policyViolations,
        emotionComparison: overrides.emotionComparison ?? baseDetail.emotionComparison,
        llmTriggers: overrides.llmTriggers ?? baseDetail.llmTriggers,
        ragCompliance: overrides.ragCompliance,
        emotionTriggers: overrides.emotionTriggers,
    }
}

const renderWithId = (id = 'int-001') =>
    render(
        <MemoryRouter initialEntries={[`/manager/inspector/${id}`]}>
            <Routes>
                <Route path="/manager/inspector/:id" element={<SessionDetail />} />
            </Routes>
        </MemoryRouter>
    )

function setupAudioElement() {
    const audio = document.querySelector('audio') as HTMLAudioElement | null
    expect(audio).not.toBeNull()
    Object.defineProperty(audio!, 'currentTime', {
        configurable: true,
        writable: true,
        value: 0,
    })
    return audio!
}

describe('SessionDetail', () => {
    beforeEach(() => {
        getInteractionDetailMock.mockReset()
        getAudioUrlMock.mockReset()
        mockPlay.mockClear()
        mockPause.mockClear()
        getAudioUrlMock.mockImplementation((id: string) => `/audio/${id}.mp3`)

        Object.defineProperty(HTMLMediaElement.prototype, 'play', {
            configurable: true,
            value: mockPlay,
        })

        Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
            configurable: true,
            value: mockPause,
        })
    })

    it('renders process adherence status and efficiency from the API response', async () => {
        getInteractionDetailMock.mockResolvedValue(
            buildDetail({
                llmTriggers: {
                    ...baseDetail.llmTriggers,
                    processAdherence: {
                        ...baseDetail.llmTriggers.processAdherence,
                        isResolved: true,
                        efficiencyScore: 9,
                    },
                },
            })
        )
        renderWithId()

        expect(await screen.findByText('Resolved')).toBeInTheDocument()
        expect(screen.getByText('9/10')).toBeInTheDocument()
        expect(screen.queryByText('Unresolved')).not.toBeInTheDocument()
    })

    it('derives policy and emotion analysis sections from llm trigger data, including optional badges', async () => {
        getInteractionDetailMock.mockResolvedValue(
            buildDetail({
                llmTriggers: {
                    ...baseDetail.llmTriggers,
                    processAdherence: {
                        ...baseDetail.llmTriggers.processAdherence,
                        missingSopSteps: ['Confirm account details', 'Explain refund timeline'],
                    },
                    nliPolicy: {
                        ...baseDetail.llmTriggers.nliPolicy,
                        policyVersion: '2026.03',
                        policyCategory: 'billing',
                        conflictResolutionApplied: true,
                    },
                    emotionShift: {
                        ...baseDetail.llmTriggers.emotionShift,
                        currentCustomerEmotion: 'grateful',
                    },
                } as any,
            })
        )
        renderWithId()

        expect(await screen.findByText('Policy Version: 2026.03')).toBeInTheDocument()
        expect(screen.getByText('Category: billing')).toBeInTheDocument()
        expect(screen.getByText('Conflict Resolved')).toBeInTheDocument()
        expect(screen.getByText('grateful')).toBeInTheDocument()
        expect(screen.getByText('Explain refund timeline')).toBeInTheDocument()
    })

    it('shows unavailable analysis states when llm trigger data is missing', async () => {
        getInteractionDetailMock.mockResolvedValue(
            buildDetail({
                llmTriggers: {
                    available: false,
                    error: 'Timed out',
                } as any,
            })
        )
        renderWithId()

        expect(await screen.findByText(/RAG compliance analysis unavailable/i)).toBeInTheDocument()
        expect(screen.getByText(/Emotion trigger analysis unavailable/i)).toBeInTheDocument()
        expect(screen.getAllByText(/Timed out/i).length).toBeGreaterThan(0)
    })

    it('seeks audio when a transcript utterance is clicked', async () => {
        getInteractionDetailMock.mockResolvedValue(buildDetail())
        renderWithId()

        expect(await screen.findByText('Customer is getting upset about the hold time.')).toBeInTheDocument()

        const audio = setupAudioElement()
        fireEvent.click(screen.getByText('Customer is getting upset about the hold time.'))
        expect(audio.currentTime).toBe(5)
        expect(mockPlay).toHaveBeenCalledTimes(1)
    })

    it('opens and closes the conversation window for a transcript utterance', async () => {
        getInteractionDetailMock.mockResolvedValue(buildDetail())
        renderWithId()

        await screen.findByText('Customer is getting upset about the hold time.')

        fireEvent.click(screen.getAllByRole('button', { name: 'Open Window' })[1])

        expect(screen.getByText('Conversation Window')).toBeInTheDocument()
        expect(screen.getByText(/Centered around selected utterance at 00:05/i)).toBeInTheDocument()

        fireEvent.click(screen.getByRole('button', { name: 'Close' }))
        expect(screen.queryByText('Conversation Window')).not.toBeInTheDocument()
    })

    it('syncs the active transcript line when the seek bar moves', async () => {
        getInteractionDetailMock.mockResolvedValue(buildDetail())
        renderWithId()

        await screen.findByText('Customer is getting upset about the hold time.')

        const audio = setupAudioElement()

        fireEvent.change(screen.getByLabelText('Audio seek'), { target: { value: '50' } })

        expect(audio.currentTime).toBe(6)
        await waitFor(() => {
            expect(document.getElementById('utterance-u2')).toHaveClass('opacity-100')
        })

        fireEvent.change(screen.getByLabelText('Audio seek'), { target: { value: '25' } })

        expect(audio.currentTime).toBe(3)
        await waitFor(() => {
            expect(document.getElementById('utterance-u1')).toHaveClass('opacity-100')
        })
    })
})
