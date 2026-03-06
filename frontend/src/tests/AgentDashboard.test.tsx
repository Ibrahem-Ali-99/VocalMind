import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { AgentDashboard } from '../app/components/agent/AgentDashboard'
import { MemoryRouter } from 'react-router'

describe('AgentDashboard', () => {
    it('renders personal performance hero card', () => {
        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        expect(screen.getByText('MY PERFORMANCE')).toBeInTheDocument()
        expect(screen.getByText('Overall Score')).toBeInTheDocument()
        expect(screen.getByText('Team Rank')).toBeInTheDocument()
    })

    it('renders specific agent stats', () => {
        render(
            <MemoryRouter>
                <AgentDashboard />
            </MemoryRouter>
        )

        // Using values from mockData or component defaults
        expect(screen.getByText('Calls Today')).toBeInTheDocument()
        expect(screen.getByText('Avg Response')).toBeInTheDocument()
        expect(screen.getByText('8')).toBeInTheDocument() // Calls today
    })
})
