'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import LoadingSpinner from '@/components/shared/LoadingSpinner'
import EmptyState from '@/components/shared/EmptyState'
import { mockCalls } from '@/api/mocks/callsData'

const getEmotionStyle = (emotion) => {
    switch (emotion) {
        case 'Satisfied':
            return 'bg-emerald-100 text-emerald-700 border-emerald-200'
        case 'Neutral':
            return 'bg-gray-100 text-gray-700 border-gray-200'
        case 'Frustrated':
            return 'bg-red-100 text-red-700 border-red-200'
        default:
            return 'bg-gray-100 text-gray-600'
    }
}

export default function CallsPage() {
    const router = useRouter()
    const [loading, setLoading] = useState(true)
    const [calls, setCalls] = useState([])
    const [currentPage, setCurrentPage] = useState(1)
    const recordsPerPage = 10

    useEffect(() => {
        const timer = setTimeout(() => {
            setCalls(mockCalls)
            setLoading(false)
        }, 500)
        return () => clearTimeout(timer)
    }, [])

    const total = calls.length
    const totalPages = Math.ceil(total / recordsPerPage)
    const paginatedCalls = calls.slice((currentPage - 1) * recordsPerPage, currentPage * recordsPerPage)

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <LoadingSpinner size="lg" />
            </div>
        )
    }

    return (
        <div className="max-w-full">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-900">All Calls</h1>
                    <p className="text-gray-500 mt-1">Total: {total.toLocaleString()} Recordings</p>
                </div>
                <button className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-all duration-200 flex items-center gap-2 text-gray-700">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Export CSV
                </button>
            </div>

            {/* Filters */}
            <div className="flex flex-col md:flex-row items-start md:items-center gap-3 mb-6 overflow-x-auto pb-2 md:pb-0">
                <button className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:border-cyan transition-all duration-200 flex items-center gap-2 text-gray-700 whitespace-nowrap">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Date Range
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </button>

                <button className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:border-cyan transition-all duration-200 flex items-center gap-2 text-gray-700 whitespace-nowrap">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    Agent: All
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </button>

                <button className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:border-cyan transition-all duration-200 flex items-center gap-2 text-gray-700 whitespace-nowrap">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Emotion: Any
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </button>

                <button className="text-cyan hover:text-cyan-dark transition-colors text-sm whitespace-nowrap">
                    Clear all
                </button>
            </div>

            {/* Table */}
            {paginatedCalls.length > 0 ? (
                <div className="bg-white rounded-xl shadow-card overflow-hidden overflow-x-auto">
                    <table className="w-full min-w-[1000px]">
                        <thead>
                            <tr className="text-left text-sm text-gray-500 border-b border-gray-100 bg-gray-50">
                                <th className="px-6 py-3 font-medium">
                                    <input type="checkbox" className="rounded border-gray-300" />
                                </th>
                                <th className="px-6 py-3 font-medium">DATE/TIME</th>
                                <th className="px-6 py-3 font-medium">AGENT</th>
                                <th className="px-6 py-3 font-medium">CUSTOMER</th>
                                <th className="px-6 py-3 font-medium">DURATION</th>
                                <th className="px-6 py-3 font-medium">EMOTION</th>
                                <th className="px-6 py-3 font-medium">
                                    AI SUMMARY
                                    <span className="ml-1 text-cyan">✦</span>
                                </th>
                                <th className="px-6 py-3 font-medium">ACTIONS</th>
                            </tr>
                        </thead>
                        <tbody>
                            {paginatedCalls.map((call) => (
                                <tr
                                    key={call.id}
                                    className="border-b border-gray-50 hover:bg-cyan/5 transition-colors cursor-pointer"
                                    onClick={() => router.push(`/session/${call.id}`)}
                                >
                                    <td className="px-6 py-4" onClick={(e) => e.stopPropagation()}>
                                        <input type="checkbox" className="rounded border-gray-300" />
                                    </td>
                                    <td className="px-6 py-4">
                                        <div>
                                            <p className="font-medium text-gray-900">{call.dateTime}</p>
                                            <p className="text-xs text-cyan">{call.timeAgo}</p>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan to-purple flex items-center justify-center text-white text-sm font-medium">
                                                {call.agent
                                                    ?.split(' ')
                                                    .map((n) => n[0])
                                                    .join('') || '?'}
                                            </div>
                                            <span className="text-gray-900 hover:text-cyan transition-colors">
                                                {call.agent}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <p className="text-gray-900">{call.customer}</p>
                                        <p className="text-xs text-gray-500">{call.phone}</p>
                                    </td>
                                    <td className="px-6 py-4 text-gray-600">{call.duration}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2.5 py-1 text-xs font-medium rounded-full border ${getEmotionStyle(call.emotion)}`}>
                                            {call.emotion}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        <p className="text-sm text-gray-600 line-clamp-2 max-w-xs">{call.summary}</p>
                                    </td>
                                    <td className="px-6 py-4" onClick={(e) => e.stopPropagation()}>
                                        <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                                            </svg>
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="px-6 py-4 border-t border-gray-100 flex flex-col sm:flex-row items-center justify-between gap-4">
                            <p className="text-sm text-gray-500 text-center sm:text-left">
                                Showing {(currentPage - 1) * recordsPerPage + 1} to{' '}
                                {Math.min(currentPage * recordsPerPage, total)} of {total.toLocaleString()} results
                            </p>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                                    disabled={currentPage === 1}
                                    className="px-3 py-1 text-sm border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Previous
                                </button>
                                <button
                                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                                    disabled={currentPage === totalPages}
                                    className="px-3 py-1 text-sm border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                <div className="bg-white rounded-xl shadow-card">
                    <EmptyState
                        title="No calls yet"
                        description="Upload call recordings or connect your call center to start analyzing."
                        icon={
                            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                            </svg>
                        }
                    />
                </div>
            )}

            {/* FAB */}
            <button
                onClick={() => router.push('/upload')}
                className="fixed bottom-8 right-8 w-14 h-14 bg-cyan text-navy rounded-full shadow-lg hover:bg-cyan-light hover:shadow-xl transition-all duration-200 flex items-center justify-center"
            >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
            </button>
        </div>
    )
}
