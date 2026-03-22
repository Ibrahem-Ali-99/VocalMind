import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ManagerDashboard } from '../app/components/manager/ManagerDashboard'
import { MemoryRouter } from 'react-router'

const { getDashboardStatsMock } = vi.hoisted(() => ({
    getDashboardStatsMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getDashboardStats: getDashboardStatsMock,
}))

describe('ManagerDashboard', () => {
    beforeEach(() => {
        getDashboardStatsMock.mockResolvedValue({
            kpis: {
                avgScore: 84,
                totalCalls: 342,
                resolutionRate: 88,
                violationCount: 12,
            },
            weeklyTrend: [{ day: 'Mon', score: 81 }],
            emotionDistribution: [{ name: 'neutral', value: 50, color: '#3B82F6' }],
            policyCompliance: [{ category: 'billing', rate: 91, color: '#10B981' }],
            agentPerformance: [
                {
                    name: 'Agent A',
                    empathy: 85,
                    policy: 84,
                    resolution: 88,
                    overallScore: 86,
                    trend: 'up',
                },
            ],
            interactions: [
                {
                    id: 'int-1',
                    agentName: 'Agent A',
                    agentId: 'agent-1',
                    date: '2026-03-21',
                    time: '10:00',
                    duration: '3:00',
                    language: 'en',
                    overallScore: 82,
                    empathyScore: 80,
                    policyScore: 78,
                    resolutionScore: 75,
                    resolved: true,
                    hasViolation: false,
                    hasOverlap: false,
                    responseTime: '1.0s',
                    status: 'completed',
                },
            ],
        })
    })

    it('renders KPI cards correctly', async () => {
        render(
            <MemoryRouter>
                <ManagerDashboard />
            </MemoryRouter>
        )

        expect(await screen.findByText('Average Score')).toBeInTheDocument()
        expect(screen.getByText('342')).toBeInTheDocument()
        expect(screen.getByText('12')).toBeInTheDocument()
    })

    it('renders dashboard section headers', async () => {
        render(
            <MemoryRouter>
                <ManagerDashboard />
            </MemoryRouter>
        )

        expect(await screen.findByText('Weekly Score Trends')).toBeInTheDocument()
        expect(screen.getByText('Emotion Distribution')).toBeInTheDocument()
        expect(screen.getByText('Agent Leaderboard')).toBeInTheDocument()
        expect(screen.getByText('Recent Interactions')).toBeInTheDocument()
    })
})
