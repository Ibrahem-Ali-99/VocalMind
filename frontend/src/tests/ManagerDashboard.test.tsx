import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ManagerDashboard } from '../app/components/manager/ManagerDashboard'
import { MemoryRouter } from 'react-router'

describe('ManagerDashboard', () => {
    it('renders KPI cards correctly', () => {
        render(
            <MemoryRouter>
                <ManagerDashboard />
            </MemoryRouter>
        )
        // Check for specific dummy values from the component
        expect(screen.getByText(/84/)).toBeInTheDocument()
        expect(screen.getByText('342')).toBeInTheDocument()
        expect(screen.getByText('12')).toBeInTheDocument()
    })

    it('renders dashboard section headers', () => {
        render(
            <MemoryRouter>
                <ManagerDashboard />
            </MemoryRouter>
        )

        expect(screen.getByText('Weekly Score Trends')).toBeInTheDocument()
        expect(screen.getByText('Emotion Distribution')).toBeInTheDocument()
        expect(screen.getByText('Agent Leaderboard')).toBeInTheDocument()
        expect(screen.getByText('Recent Interactions')).toBeInTheDocument()
    })
})
