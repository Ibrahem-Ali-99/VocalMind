import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { SessionDetail } from '../app/components/manager/SessionDetail'
import { MemoryRouter, Routes, Route } from 'react-router'

const renderWithId = (id: string = 'int-001') => render(
    <MemoryRouter initialEntries={[`/manager/inspector/${id}`]}>
        <Routes>
            <Route path="/manager/inspector/:id" element={<SessionDetail />} />
        </Routes>
    </MemoryRouter>
)

describe('SessionDetail', () => {
    it('renders call header and transcript', () => {
        renderWithId()
        expect(screen.getByText('SESSION INSPECTOR')).toBeInTheDocument()
        expect(screen.getByText('Transcript')).toBeInTheDocument()
    })

    it('displays empathy and policy scores', () => {
        renderWithId()
        expect(screen.getByText('Empathy')).toBeInTheDocument()
        expect(screen.getByText('95%')).toBeInTheDocument() // int-001
    })

    it('renders transcript utterances with speaker labels', () => {
        renderWithId()
        expect(screen.getByText(/Good morning!/)).toBeInTheDocument()
    })

    it('renders emotion events with transition labels', () => {
        renderWithId('int-001')
        expect(screen.getByText(/Customer expressed multi-day frustration/)).toBeInTheDocument()
    })

    it('shows accuracy feedback buttons after flagging an emotion event', () => {
        renderWithId('int-001')
        const flagButtons = screen.getAllByText('Flag as incorrect')
        fireEvent.click(flagButtons[0])

        expect(screen.getByText('Was this detection accurate?')).toBeInTheDocument()
    })

    it('shows confirmation after submitting feedback on an emotion event', () => {
        renderWithId('int-001')
        const flagButtons = screen.getAllByText('Flag as incorrect')
        fireEvent.click(flagButtons[0])
        fireEvent.click(screen.getByText('Accurate'))

        expect(screen.getByText(/Feedback recorded/)).toBeInTheDocument()
    })

    it('renders policy violation details with scores', () => {
        renderWithId('int-002')
        expect(screen.getByText('Hold Time Limit')).toBeInTheDocument()
        expect(screen.getByText('45%')).toBeInTheDocument()
    })

    it('shows feedback flow for policy violations', () => {
        renderWithId('int-002')
        const flagButtons = screen.getAllByText('Flag as incorrect')
        const violationFlagIndex = flagButtons.length - 1
        fireEvent.click(flagButtons[violationFlagIndex])

        expect(screen.getByText('Was this verdict correct?')).toBeInTheDocument()
        fireEvent.click(screen.getByText('Correct'))
        expect(screen.getByText(/Feedback recorded/)).toBeInTheDocument()
    })

    it('renders mid-range score color for int-005', () => {
        renderWithId('int-005')
        expect(screen.getAllByText('78%').length).toBeGreaterThan(0)
    })

    it('handles "Incorrect" feedback path for emotions', () => {
        renderWithId('int-001')
        const flagButtons = screen.getAllByText('Flag as incorrect')
        fireEvent.click(flagButtons[0])
        fireEvent.click(screen.getByText('Incorrect'))
        expect(screen.getByText(/Feedback recorded/)).toBeInTheDocument()
    })
})
