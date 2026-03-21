import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { SessionInspector } from '../app/components/manager/SessionInspector'
import { MemoryRouter } from 'react-router'

const { getInteractionsMock } = vi.hoisted(() => ({
    getInteractionsMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getInteractions: getInteractionsMock,
}))

describe('SessionInspector Component', () => {
    beforeEach(() => {
        getInteractionsMock.mockResolvedValue([
            {
                id: 'interaction-1',
                agentName: 'Sarah M.',
                date: '2026-03-20',
                time: '10:00',
                duration: '4:10',
                overallScore: 88,
                empathyScore: 90,
                policyScore: 85,
                resolutionScore: 86,
                resolved: true,
                hasViolation: false,
                status: 'completed',
            },
            {
                id: 'interaction-2',
                agentName: 'John D.',
                date: '2026-03-20',
                time: '11:30',
                duration: '5:05',
                overallScore: 79,
                empathyScore: 77,
                policyScore: 82,
                resolutionScore: 78,
                resolved: false,
                hasViolation: true,
                status: 'completed',
            },
        ])
    })

    it('renders session inspector title', async () => {
        render(
            <MemoryRouter>
                <SessionInspector />
            </MemoryRouter>
        )
        expect(await screen.findByText('Session Inspector')).toBeInTheDocument()
    })

    it('renders interaction list items', async () => {
        render(
            <MemoryRouter>
                <SessionInspector />
            </MemoryRouter>
        )
        expect(await screen.findByText('Sarah M.')).toBeInTheDocument()
        expect(screen.getByText('John D.')).toBeInTheDocument()
    })

    it('renders search input', async () => {
        render(
            <MemoryRouter>
                <SessionInspector />
            </MemoryRouter>
        )
        const searchInput = await screen.findByPlaceholderText('Search agent, date, ID...')
        expect(searchInput).toBeInTheDocument()
    })
})
