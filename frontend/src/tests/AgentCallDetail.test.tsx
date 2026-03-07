import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { AgentCallDetail } from '../app/components/agent/AgentCallDetail'
import { MemoryRouter, Routes, Route } from 'react-router'
import { mockInteractions } from '../app/data/mockData'

const renderWithId = (id: string = 'int-001') => render(
    <MemoryRouter initialEntries={[`/agent/${id}`]}>
        <Routes>
            <Route path="/agent/:id" element={<AgentCallDetail />} />
        </Routes>
    </MemoryRouter>
)

describe('AgentCallDetail', () => {
    it('renders the call header with dynamic data for int-001', () => {
        renderWithId('int-001')
        expect(screen.getByText('CALL DETAIL')).toBeInTheDocument()
        // int-001 is "2025-03-01"
        expect(screen.getByText(/2025-03-01/)).toBeInTheDocument()
    })

    it('renders score breakdown metrics for int-001', () => {
        renderWithId('int-001')
        expect(screen.getByText('Empathy')).toBeInTheDocument()
        // Use getAllByText and regex to be robust
        const elements = screen.getAllByText(/85%/)
        expect(elements.length).toBeGreaterThan(0)
        expect(screen.getByText('1.2s')).toBeInTheDocument()
    })

    it('renders coaching points from policy violations for int-002', () => {
        renderWithId('int-002') 
        expect(screen.getByText('Coaching Points')).toBeInTheDocument()
    })

    it('renders transcript section for int-001', () => {
        renderWithId('int-001')
        expect(screen.getByText('Transcript')).toBeInTheDocument()
        expect(screen.getByText(/Good morning!/)).toBeInTheDocument()
    })

    it('renders customer emotion journey for int-001', () => {
        renderWithId('int-001')
        expect(screen.getByText('Customer Emotion Journey')).toBeInTheDocument()
    })

    it('renders back navigation link', () => {
        renderWithId('int-001')
        expect(screen.getByText('Back to My Calls').closest('a')).toHaveAttribute('href', '/agent')
    })

    it('renders dynamic data for int-005 with mid-range score', () => {
        renderWithId('int-005')
        expect(screen.getByText(/2025-03-01/)).toBeInTheDocument()
        const elements = screen.getAllByText(/78%/)
        expect(elements.length).toBeGreaterThan(0)
    })
})
