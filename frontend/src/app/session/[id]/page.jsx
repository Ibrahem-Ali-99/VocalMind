'use client'

import dynamic from 'next/dynamic'
import { Suspense, use } from 'react'

const SessionInspectorClient = dynamic(
    () => import('./SessionInspectorClient'),
    { ssr: false }
)

function SessionSkeleton() {
    return (
        <div className="animate-pulse space-y-6 max-w-7xl mx-auto">
            <div className="h-8 bg-gray-200 rounded w-1/3"></div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-4">
                    <div className="h-48 bg-gray-200 rounded-xl"></div>
                    <div className="h-96 bg-gray-200 rounded-xl"></div>
                </div>
                <div className="space-y-4">
                    <div className="h-32 bg-gray-200 rounded-xl"></div>
                    <div className="h-48 bg-gray-200 rounded-xl"></div>
                    <div className="h-64 bg-gray-200 rounded-xl"></div>
                </div>
            </div>
        </div>
    )
}

export default function SessionPage({ params }) {
    const { id } = use(params)

    return (
        <Suspense fallback={<SessionSkeleton />}>
            <SessionInspectorClient sessionId={id} />
        </Suspense>
    )
}
