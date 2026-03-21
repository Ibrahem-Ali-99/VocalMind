import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { KnowledgeBase } from '../app/components/manager/KnowledgeBase'
import { MemoryRouter } from 'react-router'

const { getPoliciesMock, getFaqsMock } = vi.hoisted(() => ({
    getPoliciesMock: vi.fn(),
    getFaqsMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getPolicies: getPoliciesMock,
    getFaqs: getFaqsMock,
}))

describe('KnowledgeBase', () => {
    beforeEach(() => {
        getPoliciesMock.mockResolvedValue([
            {
                id: 'p1',
                title: 'Greeting Protocol',
                category: 'customer_service',
                content: 'content',
                preview: 'Greeting preview',
                lastUpdated: '2026-03-01',
                isActive: true,
            },
            {
                id: 'p2',
                title: 'Data Privacy Guidelines',
                category: 'compliance',
                content: 'content',
                preview: 'Privacy preview',
                lastUpdated: '2026-03-01',
                isActive: true,
            },
            {
                id: 'p3',
                title: 'Escalation Procedure',
                category: 'operations',
                content: 'content',
                preview: 'Escalation preview',
                lastUpdated: '2026-03-01',
                isActive: false,
            },
        ])

        getFaqsMock.mockResolvedValue([
            {
                id: 'f1',
                question: "How do I reset a customer's password?",
                answer: 'answer',
                preview: 'preview',
                category: 'account',
                isActive: true,
            },
            {
                id: 'f2',
                question: 'What is the refund policy?',
                answer: 'answer',
                preview: 'preview',
                category: 'billing',
                isActive: true,
            },
        ])
    })

    it('renders info banner and headers', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText(/Manage which policies and FAQ articles/)).toBeInTheDocument()
        expect(screen.getByText('Company Policies')).toBeInTheDocument()
        expect(screen.getByText('FAQ Articles')).toBeInTheDocument()
    })

    it('renders search placeholders', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByPlaceholderText('Search policies...')).toBeInTheDocument()
        expect(screen.getByPlaceholderText('Search FAQs...')).toBeInTheDocument()
    })

    it('toggles policy switch and filters policy list', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText('Greeting Protocol')).toBeInTheDocument()

        const switches = screen.getAllByRole('switch')
        const escalationSwitch = switches[2]
        expect(escalationSwitch).not.toBeChecked()
        fireEvent.click(escalationSwitch)
        expect(escalationSwitch).toBeChecked()

        const policySearch = screen.getByPlaceholderText('Search policies...')
        fireEvent.change(policySearch, { target: { value: 'Privacy' } })
        expect(screen.getByText('Data Privacy Guidelines')).toBeInTheDocument()
        expect(screen.queryByText('Greeting Protocol')).not.toBeInTheDocument()
    })
})
