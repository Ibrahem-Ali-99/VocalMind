import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { SessionInspector } from '../app/components/manager/SessionInspector'
import { MemoryRouter } from 'react-router'

describe('SessionInspector Component', () => {
    it('renders session inspector title', () => {
        render(
            <MemoryRouter>
                <SessionInspector />
            </MemoryRouter>
        )
        expect(screen.getByText('Session Inspector')).toBeInTheDocument()
    })

    it('renders interaction list items', () => {
        render(
            <MemoryRouter>
                <SessionInspector />
            </MemoryRouter>
        )
        // From real mockData.ts: Sarah M., John D.
        expect(screen.getByText('Sarah M.')).toBeInTheDocument()
        expect(screen.getByText('John D.')).toBeInTheDocument()
    })

    it('renders search input', () => {
        render(
            <MemoryRouter>
                <SessionInspector />
            </MemoryRouter>
        )
        const searchInput = screen.getByPlaceholderText('Search agent, date, ID…')
        expect(searchInput).toBeInTheDocument()
    })
})
