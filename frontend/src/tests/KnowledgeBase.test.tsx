import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { KnowledgeBase } from '../app/components/manager/KnowledgeBase'
import { MemoryRouter } from 'react-router'

const { getPoliciesMock, getFaqsMock, togglePolicyMock } = vi.hoisted(() => ({
    getPoliciesMock: vi.fn(),
    getFaqsMock: vi.fn(),
    togglePolicyMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    getPolicies: getPoliciesMock,
    getFaqs: getFaqsMock,
    togglePolicy: togglePolicyMock,
}))

describe('KnowledgeBase', () => {
    beforeEach(() => {
        getPoliciesMock.mockResolvedValue([
            {
                id: 'p1',
                documentType: 'policy',
                title: 'Greeting Protocol',
                category: 'customer_service',
                content: 'content',
                preview: 'Greeting preview',
                lastUpdated: '2026-03-01',
                isActive: true,
                usageCount: 3,
            },
            {
                id: 'p2',
                documentType: 'policy',
                title: 'Data Privacy Guidelines',
                category: 'compliance',
                content: 'content',
                preview: 'Privacy preview',
                lastUpdated: '2026-03-01',
                isActive: true,
                usageCount: 2,
            },
            {
                id: 'p3',
                documentType: 'policy',
                title: 'Escalation Procedure',
                category: 'operations',
                content: 'content',
                preview: 'Escalation preview',
                lastUpdated: '2026-03-01',
                isActive: false,
                usageCount: 0,
            },
        ])

        getFaqsMock.mockResolvedValue([
            {
                id: 'f1',
                documentType: 'faq',
                question: "How do I reset a customer's password?",
                answer: 'answer',
                preview: 'preview',
                category: 'account',
                isActive: true,
                usageCount: 4,
            },
            {
                id: 'f2',
                documentType: 'faq',
                question: 'What is the refund policy?',
                answer: 'answer',
                preview: 'preview',
                category: 'billing',
                isActive: true,
                usageCount: 1,
            },
        ])

        togglePolicyMock.mockResolvedValue({ isActive: true })
    })

    it('renders info banner and headers', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText('Knowledge Engine')).toBeInTheDocument()
        expect(screen.getByText('Guidelines')).toBeInTheDocument()
        expect(screen.getByText('SOP & Knowledge')).toBeInTheDocument()
    })

    it('renders search placeholders', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByPlaceholderText('Search guidelines...')).toBeInTheDocument()
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
        await waitFor(() => expect(escalationSwitch).toBeChecked())

        const policySearch = screen.getByPlaceholderText('Search guidelines...')
        fireEvent.change(policySearch, { target: { value: 'Privacy' } })
        expect(screen.getByText('Data Privacy Guidelines')).toBeInTheDocument()
        expect(screen.queryByText('Greeting Protocol')).not.toBeInTheDocument()
    })
})
