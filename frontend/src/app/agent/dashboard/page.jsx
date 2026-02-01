'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import LoadingSpinner from '@/components/shared/LoadingSpinner'

const StatCard = ({ label, value, suffix, trend, change, subtext, color }) => {
    const trendColor = trend === 'up' ? 'text-emerald-500' : trend === 'down' ? 'text-red-500' : 'text-gray-400'

    return (
        <div className="bg-white rounded-xl p-6 shadow-card hover:shadow-card-hover transition-all duration-200">
            <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-500">{label}</span>
                <span className={`text-sm ${trendColor}`}>{change}</span>
            </div>
            <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold text-gray-900">{value}</span>
                {suffix && <span className="text-lg text-gray-400">{suffix}</span>}
            </div>
            {subtext && <p className="text-xs text-gray-400 mt-1">{subtext}</p>}
            <div className={`h-1 mt-4 rounded-full bg-gradient-to-r ${color} opacity-60`}></div>
        </div>
    )
}

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

export default function AgentDashboardPage() {
    const router = useRouter()
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const timer = setTimeout(() => setLoading(false), 300)
        return () => clearTimeout(timer)
    }, [])

    const agentName = 'Sarah Miller'

    const stats = [
        { label: 'Calls Handled', value: '15', trend: 'up', change: '+2', color: 'from-blue-400 to-blue-600', subtext: 'Today' },
        { label: 'Avg Score', value: '4.2', suffix: '/5', trend: 'up', change: '+0.1', color: 'from-emerald-400 to-emerald-600', subtext: 'Top 10%' },
        { label: 'Active Time', value: '6h 23m', trend: 'neutral', change: '0m', color: 'from-purple-400 to-purple-600', subtext: 'Shift Progress' },
        { label: 'Resolved', value: '87%', trend: 'up', change: '+5%', color: 'from-cyan-400 to-cyan-600', subtext: 'First Contact' },
    ]

    const recentCalls = [
        { id: 'CALL-101', date: 'Today, 2:30 PM', customer: 'John Doe', duration: '5:20', score: 4.5, status: 'Resolved' },
        { id: 'CALL-102', date: 'Today, 1:15 PM', customer: 'Jane Smith', duration: '3:45', score: 3.8, status: 'Follow-up' },
        { id: 'CALL-103', date: 'Today, 11:50 AM', customer: 'Acme Corp', duration: '8:10', score: 4.8, status: 'Resolved' },
        { id: 'CALL-104', date: 'Today, 10:30 AM', customer: 'TechStart Inc', duration: '4:05', score: 4.2, status: 'Resolved' },
        { id: 'CALL-105', date: 'Yesterday, 4:45 PM', customer: 'Global West', duration: '6:30', score: 3.9, status: 'Escalated' },
    ]

    const coachingTips = [
        { category: 'Empathy', tip: 'Great job acknowledging frustration. Try using "I understand how that feels" earlier.', icon: '❤️' },
        { category: 'Response Time', tip: 'Your average hold time is slightly up. Remember to check back every 2 minutes.', icon: '⏱️' },
        { category: 'Policy', tip: 'Perfect adherence to the new authentication protocol today!', icon: '✅' },
    ]

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <LoadingSpinner size="lg" />
            </div>
        )
    }

    return (
        <div className="max-w-7xl mx-auto">
            {/* Welcome Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-900">Welcome back, {agentName}</h1>
                    <p className="text-gray-500 mt-1">Here&apos;s what happened while you were away.</p>
                </div>
                <div className="flex items-center gap-3">
                    <span className="text-sm text-gray-500 bg-white px-3 py-1.5 rounded-lg border border-gray-200">
                        Shift: 9:00 AM - 5:00 PM
                    </span>
                    <button className="px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors">
                        End Shift
                    </button>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {stats.map((stat, index) => (
                    <StatCard key={index} {...stat} />
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Content Column */}
                <div className="lg:col-span-2 space-y-6">
                    {/* My Recent Calls */}
                    <div className="bg-white rounded-xl shadow-card p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-lg font-semibold text-gray-900">My Recent Calls</h2>
                            <button
                                onClick={() => router.push('/agent/calls')}
                                className="text-sm text-cyan hover:text-cyan-dark font-medium"
                            >
                                View All History
                            </button>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="text-left text-xs text-gray-500 border-b border-gray-100">
                                        <th className="pb-3 font-medium">Date</th>
                                        <th className="pb-3 font-medium">Customer</th>
                                        <th className="pb-3 font-medium">Duration</th>
                                        <th className="pb-3 font-medium">Score</th>
                                        <th className="pb-3 font-medium">Status</th>
                                        <th className="pb-3 font-medium">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {recentCalls.map((call) => (
                                        <tr key={call.id} className="border-b border-gray-50 hover:bg-gray-50">
                                            <td className="py-3 text-sm text-gray-600">{call.date}</td>
                                            <td className="py-3 text-sm font-medium text-gray-900">{call.customer}</td>
                                            <td className="py-3 text-sm text-gray-600">{call.duration}</td>
                                            <td className="py-3">
                                                <span className={`font-medium ${call.score >= 4.0 ? 'text-emerald-600' : 'text-amber-600'}`}>
                                                    {call.score}
                                                </span>
                                            </td>
                                            <td className="py-3">
                                                <StatusBadge status={call.status} />
                                            </td>
                                            <td className="py-3">
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
                </div>

                {/* Sidebar Column */}
                <div className="space-y-6">
                    {/* Quick Actions Panel */}
                    <div className="bg-white rounded-xl shadow-card p-6">
                        <h3 className="font-semibold text-gray-900 mb-4">Quick Actions</h3>
                        <div className="space-y-3">
                            <button className="w-full flex items-center gap-3 p-3 rounded-lg bg-emerald-50 text-emerald-700 hover:bg-emerald-100 transition-colors text-sm font-medium">
                                <span className="text-xl">☕</span> Start Break
                            </button>
                            <button className="w-full flex items-center gap-3 p-3 rounded-lg bg-gray-50 text-gray-700 hover:bg-gray-100 transition-colors text-sm font-medium">
                                <span className="text-xl">🛑</span> End Shift
                            </button>
                            <button className="w-full flex items-center gap-3 p-3 rounded-lg bg-cyan/10 text-cyan-dark hover:bg-cyan/20 transition-colors text-sm font-medium">
                                <span className="text-xl">🙋‍♂️</span> Request Review
                            </button>
                        </div>
                    </div>

                    {/* AI Coaching Tips */}
                    <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-card p-6 border border-indigo-100">
                        <div className="flex items-center gap-2 mb-4">
                            <span className="text-xl">✨</span>
                            <h3 className="font-semibold text-indigo-900">AI Coaching Tips</h3>
                        </div>
                        <div className="space-y-4">
                            {coachingTips.map((tip, i) => (
                                <div key={i} className="bg-white/60 p-3 rounded-lg backdrop-blur-sm">
                                    <div className="flex items-center gap-2 text-xs font-semibold text-indigo-600 mb-1">
                                        <span>{tip.icon}</span>
                                        <span className="uppercase tracking-wider">{tip.category}</span>
                                    </div>
                                    <p className="text-sm text-indigo-900 leading-relaxed">&quot;{tip.tip}&quot;</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
