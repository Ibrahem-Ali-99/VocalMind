import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router'

import { ManagerAssistant } from '../app/components/manager/ManagerAssistant'

const { sendAssistantQueryMock, getAssistantHistoryMock } = vi.hoisted(() => ({
    sendAssistantQueryMock: vi.fn(),
    getAssistantHistoryMock: vi.fn(),
}))

vi.mock('../app/services/api', () => ({
    sendAssistantQuery: sendAssistantQueryMock,
    getAssistantHistory: getAssistantHistoryMock,
}))

describe('ManagerAssistant', () => {
    beforeEach(() => {
        sendAssistantQueryMock.mockReset()
        getAssistantHistoryMock.mockReset()
        getAssistantHistoryMock.mockResolvedValue([])
        sendAssistantQueryMock.mockResolvedValue({
            id: 'ai-1',
            type: 'ai',
            content: 'Mocked assistant reply',
            mode: 'chat',
            success: true,
            data: [],
        })
    })

    it('submits a typed query with Enter and clears the input after a response', async () => {
        render(
            <MemoryRouter>
                <ManagerAssistant />
            </MemoryRouter>
        )

        const input = screen.getByPlaceholderText(/Ask about scores, violations, agent trends/) as HTMLInputElement
        fireEvent.change(input, { target: { value: 'Test question' } })
        fireEvent.keyDown(input, { key: 'Enter' })

        expect(sendAssistantQueryMock).toHaveBeenCalledWith('Test question')
        expect(await screen.findByText('Mocked assistant reply')).toBeInTheDocument()
        expect(input.value).toBe('')
    })

    it('renders saved history with structured result data', async () => {
        getAssistantHistoryMock.mockResolvedValue([
            {
                id: 'ai-history',
                type: 'ai',
                content: 'Previous insight',
                mode: 'chat',
                success: true,
                data: [{ name: 'Sarah M.', score: 92 }],
                execution_time: '120ms',
            },
        ])

        render(
            <MemoryRouter>
                <ManagerAssistant />
            </MemoryRouter>
        )

        expect(await screen.findByText('Previous insight')).toBeInTheDocument()
        expect(screen.getByRole('table')).toBeInTheDocument()
        expect(screen.getByText('Sarah M.')).toBeInTheDocument()
        expect(screen.getByText('Executed in 120ms')).toBeInTheDocument()
    })

    it('renders the service fallback message when the request fails', async () => {
        sendAssistantQueryMock.mockRejectedValue(new Error('backend offline'))

        render(
            <MemoryRouter>
                <ManagerAssistant />
            </MemoryRouter>
        )

        const input = screen.getByPlaceholderText(/Ask about scores, violations, agent trends/)
        fireEvent.change(input, { target: { value: 'Test question' } })
        fireEvent.keyDown(input, { key: 'Enter' })

        expect(await screen.findByText(/having trouble connecting to the service/i)).toBeInTheDocument()
    })

    it('ignores whitespace-only input instead of sending an empty assistant request', async () => {
        render(
            <MemoryRouter>
                <ManagerAssistant />
            </MemoryRouter>
        )

        const input = screen.getByPlaceholderText(/Ask about scores, violations, agent trends/)
        fireEvent.change(input, { target: { value: '   ' } })
        fireEvent.keyDown(input, { key: 'Enter' })

        expect(sendAssistantQueryMock).not.toHaveBeenCalled()
        expect(input).toHaveValue('   ')
    })

    it('submits a suggested query, disables controls while loading, and then renders the reply', async () => {
        let resolveResponse: ((value: any) => void) | undefined
        sendAssistantQueryMock.mockReturnValue(
            new Promise((resolve) => {
                resolveResponse = resolve
            })
        )

        render(
            <MemoryRouter>
                <ManagerAssistant />
            </MemoryRouter>
        )

        fireEvent.click(screen.getByRole('button', { name: 'List all policy violations' }))

        expect(sendAssistantQueryMock).toHaveBeenCalledWith('List all policy violations')
        expect(screen.getByPlaceholderText(/Ask about scores, violations, agent trends/)).toBeDisabled()
        expect(screen.getByRole('button', { name: 'Who are the top 5 agents by overall score?' })).toBeDisabled()

        resolveResponse?.({
            id: 'ai-suggested',
            type: 'ai',
            content: 'There are 4 open policy violations.',
            mode: 'chat',
            success: true,
            data: [],
        })

        expect(await screen.findByText('There are 4 open policy violations.')).toBeInTheDocument()
    })

    it('renders generated sql and formatted numeric result values from assistant history', async () => {
        getAssistantHistoryMock.mockResolvedValue([
            {
                id: 'ai-history',
                type: 'ai',
                content: 'Historical answer',
                mode: 'chat',
                success: true,
                data: [{ agent_name: 'Sarah M.', avg_score: 91.25 }],
                sql: 'SELECT agent_name, avg_score FROM leaderboard',
                execution_time: '95ms',
            },
        ])

        render(
            <MemoryRouter>
                <ManagerAssistant />
            </MemoryRouter>
        )

        expect(await screen.findByText('Historical answer')).toBeInTheDocument()
        expect(screen.getByText('avg score')).toBeInTheDocument()
        expect(screen.getByText('91.3')).toBeInTheDocument()
        expect(screen.getByText('Show generated SQL')).toBeInTheDocument()
        expect(screen.getByText(/SELECT agent_name, avg_score FROM leaderboard/i)).toBeInTheDocument()
    })
})
