import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { SessionDetail } from '../app/components/manager/SessionDetail'
import { MemoryRouter } from 'react-router'

describe('SessionDetail', () => {
    it('renders call header and transcript', () => {
        render(
            <MemoryRouter>
                <SessionDetail />
            </MemoryRouter>
        )

        expect(screen.getByText('SESSION INSPECTOR')).toBeInTheDocument()
        expect(screen.getByText('Transcript')).toBeInTheDocument()
        expect(screen.getByText('Emotion Events')).toBeInTheDocument()
        expect(screen.getByText('Policy Violations')).toBeInTheDocument()
    })

    it('displays empathy and policy scores', () => {
        render(
            <MemoryRouter>
                <SessionDetail />
            </MemoryRouter>
        )

        expect(screen.getByText('Empathy')).toBeInTheDocument()
        expect(screen.getByText('Policy')).toBeInTheDocument()
        expect(screen.getByText('Resolution')).toBeInTheDocument()
    })
})
