'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import LoadingSpinner from '@/components/shared/LoadingSpinner'
import EmptyState from '@/components/shared/EmptyState'
import { mockDashboardStats, mockHighPriorityCalls } from '@/api/mocks/dashboardData'

const TrendIcon = ({ trend }) => {
    if (trend === 'up') {
        return (
            <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
        )
    }
    if (trend === 'down') {
        return (
            <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6" />
            </svg>
        )
    }
    return <span className="text-gray-400">—</span>
}

const PriorityBadge = ({ priority }) => {
    const config = {
        Critical: { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200', icon: '!' },
        High: { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200', icon: '⚠' },
        Medium: { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-200', icon: '○' },
    }
    const style = config[priority] || config['Medium']
    return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded border ${style.bg} ${style.text} ${style.border}`}>
            <span>{style.icon}</span>
            {priority}
        </span>
    )
}

export default function DashboardPage() {
    const router = useRouter()
    const [loading, setLoading] = useState(true)
    const [stats, setStats] = useState([])
    const [highPriorityCalls, setHighPriorityCalls] = useState([])
    const [priorityFilter, setPriorityFilter] = useState('All')
    const [issueTypeFilter, setIssueTypeFilter] = useState('All')
    const [activeView, setActiveView] = useState('overview')
    const [autoRefresh, setAutoRefresh] = useState(false)

    useEffect(() => {
        // Simulate API call
        const timer = setTimeout(() => {
            setStats(mockDashboardStats)
            setHighPriorityCalls(mockHighPriorityCalls)
            setLoading(false)
        }, 500)
        return () => clearTimeout(timer)
    }, [])

    const filteredCalls = highPriorityCalls.filter((call) => {
        const matchesPriority = priorityFilter === 'All' || call.priority === priorityFilter
        const matchesIssue = issueTypeFilter === 'All' || call.issueType === issueTypeFilter
        return matchesPriority && matchesIssue
    })

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <LoadingSpinner size="lg" />
            </div>
        )
    }

    return (
        <div className="max-w-7xl mx-auto">
            {/* View Tabs */}
            <div className="mb-6">
                <div className="border-b border-gray-200">
                    <nav className="-mb-px flex gap-8">
                        <button
                            onClick={() => setActiveView('overview')}
                            className={`pb-4 px-1 border-b-2 font-medium text-sm transition-colors ${activeView === 'overview'
                                ? 'border-cyan text-cyan'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                        >
                            Overview
                        </button>
                        <button
                            onClick={() => setActiveView('analytics')}
                            className={`pb-4 px-1 border-b-2 font-medium text-sm transition-colors ${activeView === 'analytics'
                                ? 'border-cyan text-cyan'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                        >
                            Analytics
                        </button>
                    </nav>
                </div>
            </div>

            {activeView === 'analytics' ? (
                <div className="bg-white rounded-xl shadow-card p-12">
                    <div className="text-center">
                        <svg className="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">Analytics Dashboard</h3>
                        <p className="text-gray-500">Advanced analytics and insights coming soon</p>
                    </div>
                </div>
            ) : (
                <>
                    {/* Welcome Header */}
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-2xl font-semibold text-gray-900">Welcome back</h1>
                            <p className="text-gray-500 mt-1">Here&apos;s what&apos;s happening with your team today.</p>
                        </div>
                        <div className="flex items-center gap-3">
                            <button className="px-4 py-2 text-gray-600 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-all duration-200 flex items-center gap-2">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                Today
                            </button>
                            <button className="px-4 py-2 bg-navy text-white rounded-lg hover:bg-navy-light transition-all duration-200 flex items-center gap-2">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                </svg>
                                Export Report
                            </button>
                        </div>
                    </div>

                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                        {stats.map((stat, index) => (
                            <div key={index} className="bg-white rounded-xl p-6 shadow-card hover:shadow-card-hover transition-all duration-200">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm text-gray-500">{stat.label}</span>
                                    <div className="flex items-center gap-1 text-sm">
                                        <TrendIcon trend={stat.trend} />
                                        <span className={stat.trend === 'up' ? 'text-emerald-500' : stat.trend === 'down' ? 'text-red-500' : 'text-gray-400'}>
                                            {stat.change}
                                        </span>
                                    </div>
                                </div>
                                <div className="flex items-baseline gap-1">
                                    <span className="text-3xl font-bold text-gray-900">{stat.value}</span>
                                    {stat.suffix && <span className="text-lg text-gray-400">{stat.suffix}</span>}
                                </div>
                                {stat.subtext && <p className="text-xs text-gray-400 mt-1">{stat.subtext}</p>}
                                <div className={`h-1 mt-4 rounded-full bg-gradient-to-r ${stat.color} opacity-60`}></div>
                            </div>
                        ))}
                    </div>

                    {/* Flagged Calls Table */}
                    <div className="bg-white rounded-xl shadow-card">
                        <div className="px-6 py-4 border-b border-gray-100">
                            <div className="flex items-center justify-between mb-3">
                                <div>
                                    <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                                        <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                        </svg>
                                        Flagged Calls - Requires Review
                                    </h2>
                                    <p className="text-sm text-gray-500">Calls flagged for review</p>
                                </div>
                                <div className="flex items-center gap-3">
                                    <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={autoRefresh}
                                            onChange={(e) => setAutoRefresh(e.target.checked)}
                                            className="rounded border-gray-300 text-cyan focus:ring-cyan"
                                        />
                                        Auto-refresh
                                    </label>
                                    <button
                                        onClick={() => router.push('/manager/calls')}
                                        className="text-sm text-cyan hover:text-cyan-dark transition-colors flex items-center gap-1 cursor-pointer"
                                    >
                                        View All
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                        </svg>
                                    </button>
                                </div>
                            </div>

                            {/* Filters */}
                            <div className="flex items-center gap-3">
                                <select
                                    value={priorityFilter}
                                    onChange={(e) => setPriorityFilter(e.target.value)}
                                    className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan/50"
                                >
                                    <option value="All">All Priorities</option>
                                    <option value="Critical">Critical</option>
                                    <option value="High">High</option>
                                    <option value="Medium">Medium</option>
                                </select>

                                <select
                                    value={issueTypeFilter}
                                    onChange={(e) => setIssueTypeFilter(e.target.value)}
                                    className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan/50"
                                >
                                    <option value="All">All Issues</option>
                                    <option value="Compliance">Compliance</option>
                                    <option value="Sentiment">Sentiment</option>
                                    <option value="Keyword">Keyword Match</option>
                                    <option value="Manual">Manual Flag</option>
                                </select>

                                {(priorityFilter !== 'All' || issueTypeFilter !== 'All') && (
                                    <button
                                        onClick={() => {
                                            setPriorityFilter('All')
                                            setIssueTypeFilter('All')
                                        }}
                                        className="text-sm text-cyan hover:text-cyan-dark"
                                    >
                                        Clear filters
                                    </button>
                                )}
                            </div>
                        </div>

                        {filteredCalls.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="text-left text-xs text-gray-500 border-b border-gray-100 bg-gray-50">
                                            <th className="px-6 py-3 font-medium">PRIORITY</th>
                                            <th className="px-6 py-3 font-medium">CALL ID</th>
                                            <th className="px-6 py-3 font-medium">AGENT</th>
                                            <th className="px-6 py-3 font-medium">CUSTOMER</th>
                                            <th className="px-6 py-3 font-medium">ISSUE TYPE</th>
                                            <th className="px-6 py-3 font-medium">TIMESTAMP</th>
                                            <th className="px-6 py-3 font-medium">ACTIONS</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredCalls.map((call) => (
                                            <tr key={call.id} className="border-b border-gray-50 hover:bg-cyan/5 transition-colors group cursor-pointer" onClick={() => router.push(`/session/${call.id}`)}>
                                                <td className="px-6 py-4">
                                                    <PriorityBadge priority={call.priority} />
                                                </td>
                                                <td className="px-6 py-4">
                                                    <button
                                                        onClick={() => router.push(`/session/${call.id}`)}
                                                        className="font-medium text-cyan hover:text-cyan-dark cursor-pointer"
                                                    >
                                                        #{call.id}
                                                    </button>
                                                </td>
                                                <td className="px-6 py-4 text-gray-600">{call.agent}</td>
                                                <td className="px-6 py-4 text-gray-600">{call.customer}</td>
                                                <td className="px-6 py-4">
                                                    <span className="px-2 py-0.5 text-xs font-medium bg-purple/10 text-purple rounded">
                                                        {call.issueType}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-gray-500 text-sm">{call.timestamp}</td>
                                                <td className="px-6 py-4">
                                                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                router.push(`/session/${call.id}`);
                                                            }}
                                                            className="px-3 py-1 text-xs font-medium text-cyan border border-cyan rounded hover:bg-cyan hover:text-navy transition-all cursor-pointer"
                                                        >
                                                            Review
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <EmptyState
                                title="No flagged calls"
                                description={
                                    priorityFilter !== 'All' || issueTypeFilter !== 'All'
                                        ? 'No calls match your filters.'
                                        : 'When calls are flagged for review, they will appear here.'
                                }
                                icon={
                                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                                    </svg>
                                }
                            />
                        )}
                    </div>
                </>
            )}
        </div>
    )
}
