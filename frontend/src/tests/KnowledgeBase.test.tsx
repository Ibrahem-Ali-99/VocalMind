import React from 'react'
import { render, screen } from '@testing-library/react'
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
})
