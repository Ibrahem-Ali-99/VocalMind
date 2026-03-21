import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { AgentDashboard } from '../app/components/agent/AgentDashboard'
import { MemoryRouter } from 'react-router'

const { getAgentsMock, getAgentProfileMock } = vi.hoisted(() => ({
    getAgentsMock: vi.fn(),
    getAgentProfileMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getAgents: getAgentsMock,
    getAgentProfile: getAgentProfileMock,
}))

describe('AgentDashboard', () => {
    beforeEach(() => {
        getAgentsMock.mockResolvedValue([{ id: 'agent-1', name: 'Agent A', role: 'Support Agent' }])
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
            avgResponseTime: '1.1',
            trend: 'up',
            weeklyTrend: [{ day: 'Mon', score: 80 }],
            recentCalls: [],
        })
    })

    it('renders personal performance hero card', async () => {
        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        expect(await screen.findByText('MY PERFORMANCE')).toBeInTheDocument()
        expect(screen.getAllByText('Overall Score').length).toBeGreaterThan(0)
        expect(screen.getByText('Team Rank')).toBeInTheDocument()
    })

    it('renders specific agent stats', async () => {
        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        expect(await screen.findByText('Calls Today')).toBeInTheDocument()
        expect(screen.getByText('Avg Response')).toBeInTheDocument()
        expect(screen.getByText('processed calls')).toBeInTheDocument()
    })
})
