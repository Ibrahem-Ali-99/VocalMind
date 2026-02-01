'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import LoadingSpinner from '@/components/shared/LoadingSpinner'

const StatusBadge = ({ status }) => {
    const config = {
        Resolved: 'bg-emerald-100 text-emerald-700',
        'Follow-up': 'bg-amber-100 text-amber-700',
        Escalated: 'bg-red-100 text-red-700',
    }
    return (
        <span className={`px-2 py-0.5 text-xs font-medium rounded ${config[status] || 'bg-gray-100 text-gray-600'}`}>
            {status}
        </span>
    )
}

export default function AgentCallsPage() {
    const router = useRouter()
    const [loading, setLoading] = useState(true)
    const [currentPage, setCurrentPage] = useState(1)

    useEffect(() => {
        const timer = setTimeout(() => setLoading(false), 300)
        return () => clearTimeout(timer)
    }, [])

    const calls = [
        { id: 'CALL-101', date: 'Today, 2:30 PM', customer: 'John Doe', duration: '5:20', score: 4.5, status: 'Resolved' },
        { id: 'CALL-102', date: 'Today, 1:15 PM', customer: 'Jane Smith', duration: '3:45', score: 3.8, status: 'Follow-up' },
        { id: 'CALL-103', date: 'Today, 11:50 AM', customer: 'Acme Corp', duration: '8:10', score: 4.8, status: 'Resolved' },
        { id: 'CALL-104', date: 'Today, 10:30 AM', customer: 'TechStart Inc', duration: '4:05', score: 4.2, status: 'Resolved' },
        { id: 'CALL-105', date: 'Yesterday, 4:45 PM', customer: 'Global West', duration: '6:30', score: 3.9, status: 'Escalated' },
        { id: 'CALL-106', date: 'Yesterday, 3:20 PM', customer: 'Sarah Connor', duration: '2:15', score: 4.6, status: 'Resolved' },
        { id: 'CALL-107', date: 'Yesterday, 11:00 AM', customer: 'Mike Ross', duration: '5:00', score: 4.1, status: 'Resolved' },
        { id: 'CALL-108', date: 'Yesterday, 9:30 AM', customer: 'Harvey Specter', duration: '3:30', score: 4.0, status: 'Resolved' },
    ]

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <LoadingSpinner size="lg" />
            </div>
        )
    }

    return (
        <div className="max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-900">My Call History</h1>
                    <p className="text-gray-500 mt-1">{calls.length} calls</p>
                </div>
                <div className="flex items-center gap-3">
                    <select className="px-3 py-2 border border-gray-200 rounded-lg text-sm">
                        <option>Last 7 days</option>
                        <option>Last 30 days</option>
                        <option>All time</option>
                    </select>
                </div>
            </div>

            <div className="bg-white rounded-xl shadow-card overflow-hidden">
                <table className="w-full">
                    <thead>
                        <tr className="text-left text-xs text-gray-500 border-b border-gray-100 bg-gray-50">
                            <th className="px-6 py-3 font-medium">Date</th>
                            <th className="px-6 py-3 font-medium">Customer</th>
                            <th className="px-6 py-3 font-medium">Duration</th>
                            <th className="px-6 py-3 font-medium">Score</th>
                            <th className="px-6 py-3 font-medium">Status</th>
                            <th className="px-6 py-3 font-medium">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {calls.map((call) => (
                            <tr key={call.id} className="border-b border-gray-50 hover:bg-gray-50">
                                <td className="px-6 py-4 text-sm text-gray-600">{call.date}</td>
                                <td className="px-6 py-4 text-sm font-medium text-gray-900">{call.customer}</td>
                                <td className="px-6 py-4 text-sm text-gray-600">{call.duration}</td>
                                <td className="px-6 py-4">
                                    <span className={`font-medium ${call.score >= 4.0 ? 'text-emerald-600' : 'text-amber-600'}`}>
                                        {call.score}
                                    </span>
                                </td>
                                <td className="px-6 py-4">
                                    <StatusBadge status={call.status} />
                                </td>
                                <td className="px-6 py-4">
                                    <button
                                        onClick={() => router.push(`/session/${call.id}`)}
                                        className="text-cyan hover:text-cyan-dark text-sm font-medium"
                                    >
                                        Review
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
