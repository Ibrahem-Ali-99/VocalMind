import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router'

import { KnowledgeBase } from '../app/components/manager/KnowledgeBase'

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

    it('filters policies and lets the visible policy be toggled', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText('Greeting Protocol')).toBeInTheDocument()

        fireEvent.change(screen.getByPlaceholderText('Search policies...'), { target: { value: 'Escalation' } })

        expect(screen.getByText('Escalation Procedure')).toBeInTheDocument()
        expect(screen.queryByText('Greeting Protocol')).not.toBeInTheDocument()

        const [visibleSwitch] = screen.getAllByRole('switch')
        expect(visibleSwitch).not.toBeChecked()

        fireEvent.click(visibleSwitch)

        expect(visibleSwitch).toBeChecked()
    })

    it('filters faq articles independently from the policy list', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText("How do I reset a customer's password?")).toBeInTheDocument()

        fireEvent.change(screen.getByPlaceholderText('Search FAQs...'), { target: { value: 'refund' } })

        expect(screen.getByText('What is the refund policy?')).toBeInTheDocument()
        expect(screen.queryByText("How do I reset a customer's password?")).not.toBeInTheDocument()
    })

    it('shows a loading state until both knowledge sources finish loading', async () => {
        let resolvePolicies: ((value: any[]) => void) | undefined
        let resolveFaqs: ((value: any[]) => void) | undefined

        getPoliciesMock.mockReturnValue(
            new Promise((resolve) => {
                resolvePolicies = resolve
            })
        )
        getFaqsMock.mockReturnValue(
            new Promise((resolve) => {
                resolveFaqs = resolve
            })
        )

        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(screen.getByText('Loading knowledge base...')).toBeInTheDocument()

        resolvePolicies?.([
            {
                id: 'p1',
                title: 'Greeting Protocol',
                category: 'customer_service',
                content: 'content',
                preview: 'Greeting preview',
                lastUpdated: '2026-03-01',
                isActive: true,
            },
        ])
        resolveFaqs?.([
            {
                id: 'f1',
                question: "How do I reset a customer's password?",
                answer: 'answer',
                preview: 'preview',
                category: 'account',
                isActive: true,
            },
        ])

        expect(await screen.findByText('Greeting Protocol')).toBeInTheDocument()
    })

    it('shows the shared error state when one knowledge source fails', async () => {
        getPoliciesMock.mockRejectedValue(new Error('policies unavailable'))
        getFaqsMock.mockResolvedValue([])

        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText('Failed to load knowledge base')).toBeInTheDocument()
        expect(screen.getByText('policies unavailable')).toBeInTheDocument()
    })

    it('toggles the filtered faq article without affecting the policy results', async () => {
        render(
            <MemoryRouter>
                <KnowledgeBase />
            </MemoryRouter>
        )

        expect(await screen.findByText('Greeting Protocol')).toBeInTheDocument()

        fireEvent.change(screen.getByPlaceholderText('Search FAQs...'), { target: { value: 'refund' } })

        const switches = screen.getAllByRole('switch')
        const faqSwitch = switches[switches.length - 1]

        expect(screen.getByText('What is the refund policy?')).toBeInTheDocument()
        expect(screen.getByText('Greeting Protocol')).toBeInTheDocument()
        expect(faqSwitch).toBeChecked()

        fireEvent.click(faqSwitch)

        expect(faqSwitch).not.toBeChecked()
    })
})
