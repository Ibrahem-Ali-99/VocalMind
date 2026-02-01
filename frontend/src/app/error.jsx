'use client'

import { useEffect } from 'react'
import ErrorState from '@/components/shared/ErrorState'

export default function Error({ error, reset }) {
    useEffect(() => {
        // Log the error to an error reporting service
        console.error(error)
    }, [error])

    return (
        <div className="flex h-screen items-center justify-center bg-gray-50">
            <div className="bg-white p-6 rounded-xl shadow-card w-full max-w-lg">
                <ErrorState
                    title="Something went wrong!"
                    message={error.message || "We encountered an unexpected error."}
                    onRetry={reset}
                />
            </div>
        </div>
    )
}
