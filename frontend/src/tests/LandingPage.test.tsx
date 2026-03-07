import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { LandingPage } from '../app/components/LandingPage'
import { MemoryRouter } from 'react-router'

describe('LandingPage', () => {
    it('renders the brand title and subtitle', () => {
        render(
            <MemoryRouter>
                <LandingPage />
            </MemoryRouter>
        )

        expect(screen.getByText('VocalMind')).toBeInTheDocument()
        expect(screen.getByText('AI-Powered Call Centre Evaluation Platform')).toBeInTheDocument()
    })

    it('renders both portal cards with descriptions', () => {
        render(
            <MemoryRouter>
                <LandingPage />
            </MemoryRouter>
        )

        expect(screen.getByText('Manager Portal')).toBeInTheDocument()
        expect(screen.getByText('Agent Portal')).toBeInTheDocument()
        expect(screen.getByText('Full org access')).toBeInTheDocument()
        expect(screen.getByText('Personal view only')).toBeInTheDocument()
    })

    it('renders navigation links to portal routes', () => {
        render(
            <MemoryRouter>
                <LandingPage />
            </MemoryRouter>
        )

        expect(screen.getByText('Enter Manager Portal →').closest('a')).toHaveAttribute('href', '/manager')
        expect(screen.getByText('Enter Agent Portal →').closest('a')).toHaveAttribute('href', '/agent')
    })

    it('renders feature descriptions for each portal', () => {
        render(
            <MemoryRouter>
                <LandingPage />
            </MemoryRouter>
        )

        expect(screen.getByText('Comprehensive dashboard with org-wide KPIs')).toBeInTheDocument()
        expect(screen.getByText('Session Inspector with emotion detection')).toBeInTheDocument()
        expect(screen.getByText('Personal performance metrics and trends')).toBeInTheDocument()
        expect(screen.getByText('Individual call transcripts and analysis')).toBeInTheDocument()
    })

    it('renders the footer note', () => {
        render(
            <MemoryRouter>
                <LandingPage />
            </MemoryRouter>
        )

        expect(screen.getByText('Each portal provides a tailored experience based on your role')).toBeInTheDocument()
    })
})
