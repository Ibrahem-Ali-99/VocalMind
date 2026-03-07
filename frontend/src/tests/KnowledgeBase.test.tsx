import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { KnowledgeBase } from '../app/components/manager/KnowledgeBase'
import { MemoryRouter } from 'react-router'

describe('KnowledgeBase', () => {
    it('renders info banner and headers', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(screen.getByText(/Manage which policies and FAQ articles/)).toBeInTheDocument()
        expect(screen.getByText('Company Policies')).toBeInTheDocument()
        expect(screen.getByText('FAQ Articles')).toBeInTheDocument()
    })

    it('renders search placeholders', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(screen.getByPlaceholderText('Search policies...')).toBeInTheDocument()
        expect(screen.getByPlaceholderText('Search FAQs...')).toBeInTheDocument()
    })

    it('renders policy and FAQ items with active/inactive status', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(screen.getByText('Greeting Protocol')).toBeInTheDocument()
        expect(screen.getByText('Data Privacy Guidelines')).toBeInTheDocument()
        expect(screen.getByText('How do I reset a customer\'s password?')).toBeInTheDocument()
    })

    it('toggles a policy switch', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        // Escalation Procedure starts as inactive; find its switch
        const switches = screen.getAllByRole('switch')
        // Third switch corresponds to the third policy (Escalation Procedure, isActive: false)
        const escalationSwitch = switches[2]
        expect(escalationSwitch).not.toBeChecked()

        fireEvent.click(escalationSwitch)
        expect(escalationSwitch).toBeChecked()
    })

    it('toggles a FAQ switch', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        const switches = screen.getAllByRole('switch')
        // FAQ switches come after policy switches (3 policies + index 0 = switches[3])
        const faqSwitch = switches[3]
        expect(faqSwitch).toBeChecked()

        fireEvent.click(faqSwitch)
        expect(faqSwitch).not.toBeChecked()
    })

    it('filters policies by search input', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        const policySearch = screen.getByPlaceholderText('Search policies...')
        fireEvent.change(policySearch, { target: { value: 'Privacy' } })

        expect(screen.getByText('Data Privacy Guidelines')).toBeInTheDocument()
        expect(screen.queryByText('Greeting Protocol')).not.toBeInTheDocument()
    })

    it('filters FAQs by search input', () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        const faqSearch = screen.getByPlaceholderText('Search FAQs...')
        fireEvent.change(faqSearch, { target: { value: 'refund' } })

        expect(screen.getByText('What is the refund policy?')).toBeInTheDocument()
        expect(screen.queryByText('How do I reset a customer\'s password?')).not.toBeInTheDocument()
    })
})

