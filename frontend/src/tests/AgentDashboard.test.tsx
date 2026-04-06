import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Route, Routes } from 'react-router'

import { AgentDashboard } from '../app/components/agent/AgentDashboard'

const { getAgentsMock, getAgentProfileMock, getUserMeMock } = vi.hoisted(() => ({
    getAgentsMock: vi.fn(),
    getAgentProfileMock: vi.fn(),
    getUserMeMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getAgents: getAgentsMock,
    getAgentProfile: getAgentProfileMock,
    getUserMe: getUserMeMock,
}))

describe('AgentDashboard', () => {
    beforeEach(() => {
        getAgentsMock.mockReset()
        getAgentProfileMock.mockReset()
        getUserMeMock.mockReset()
    })

    it('loads the authenticated agent profile when no route param is present', async () => {
        getUserMeMock.mockResolvedValue({ id: 'agent-1', role: 'agent' })
        getAgentProfileMock.mockResolvedValue({
            id: 'agent-1',
            name: 'Agent A',
            role: 'Support Agent',
            totalCalls: 20,
            callsThisWeek: 8,
            teamRank: 2,
            avgScore: 87,
            overallScore: 88,
            empathyScore: 90,
            policyScore: 86,
            resolutionScore: 85,
            resolutionRate: 92,
            avgResponseTime: '1.1s',
            trend: 'up',
            weeklyTrend: [{ day: 'Mon', score: 80 }],
            recentCalls: [],
        })

        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        await waitFor(() => {
            expect(getAgentProfileMock).toHaveBeenCalledWith('agent-1')
        })
        expect(getAgentsMock).not.toHaveBeenCalled()
        expect(await screen.findByText('1.1s')).toBeInTheDocument()
    })

    it('shows a clear error when there are no agents to load', async () => {
        getUserMeMock.mockRejectedValue(new Error('no session'))
        getAgentsMock.mockResolvedValue([])

        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        expect(await screen.findByText('Failed to load agent data')).toBeInTheDocument()
        expect(screen.getByText('No agents found in the database')).toBeInTheDocument()
    })

    it('renders recent calls from the loaded agent profile', async () => {
        getUserMeMock.mockResolvedValue({ id: 'agent-1', role: 'agent' })
        getAgentProfileMock.mockResolvedValue({
            id: 'agent-1',
            name: 'Agent A',
            role: 'Support Agent',
            totalCalls: 20,
            callsThisWeek: 8,
            teamRank: 2,
            avgScore: 87,
            overallScore: 88,
            empathyScore: 90,
            policyScore: 86,
            resolutionScore: 85,
            resolutionRate: 92,
            avgResponseTime: '1.1s',
            trend: 'up',
            weeklyTrend: [{ day: 'Mon', score: 80 }],
            recentCalls: [
                {
                    id: 'call-1',
                    time: '10:00 AM',
                    score: 72,
                    duration: '3:10',
                    language: 'en',
                    resolved: false,
                    hasReview: true,
                },
            ],
        })

        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        const recentCall = await screen.findByRole('link', { name: /10:00 AM/i })
        expect(recentCall).toHaveAttribute('href', '/agent/calls/call-1')
        expect(screen.getByText('Review needed')).toBeInTheDocument()
    })

    it('uses the route agent id directly when one is present', async () => {
        getUserMeMock.mockResolvedValue({ id: 'agent-self', role: 'agent' })
        getAgentProfileMock.mockResolvedValue({
            id: 'agent-99',
            name: 'Agent Route',
            role: 'Support Agent',
            totalCalls: 18,
            callsThisWeek: 6,
            teamRank: 4,
            avgScore: 82,
            overallScore: 84,
            empathyScore: 83,
            policyScore: 81,
            resolutionScore: 80,
            resolutionRate: 86,
            avgResponseTime: '1.4s',
            trend: 'up',
            weeklyTrend: [{ day: 'Mon', score: 82 }],
            recentCalls: [],
        })

        render(
            <MemoryRouter initialEntries={['/agent/agent-99']}>
                <Routes>
                    <Route path="/agent/:agentId" element={<AgentDashboard />} />
                </Routes>
            </MemoryRouter>
        )

        await waitFor(() => {
            expect(getAgentProfileMock).toHaveBeenCalledWith('agent-99')
        })
        expect(getAgentsMock).not.toHaveBeenCalled()
        expect(await screen.findByText('Agent Route')).toBeInTheDocument()
    })

    it('shows an error state when the profile request fails', async () => {
        getUserMeMock.mockResolvedValue({ id: 'agent-self', role: 'agent' })
        getAgentProfileMock.mockRejectedValue(new Error('profile unavailable'))

        render(
            <MemoryRouter initialEntries={['/agent/agent-77']}>
                <Routes>
                    <Route path="/agent/:agentId" element={<AgentDashboard />} />
                </Routes>
            </MemoryRouter>
        )

        expect(await screen.findByText('Failed to load agent data')).toBeInTheDocument()
        expect(screen.getByText('profile unavailable')).toBeInTheDocument()
    })
})
