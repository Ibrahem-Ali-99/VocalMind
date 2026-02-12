'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import PageHeader from '@/components/shared/PageHeader'
import Table from '@/components/shared/Table'
import StatusBadge from '@/components/shared/StatusBadge'
import StarRating from '@/components/shared/StarRating'

// Mock team members data
const mockTeamMembers = [
    {
        id: 1,
        name: 'Sarah Johnson',
        role: 'Manager',
        email: 'sarah.johnson@vocalmind.io',
        activeCalls: 0,
        avgScore: 4.8,
        status: 'active',
        avatar: 'SJ',
    },
    {
        id: 2,
        name: 'Ahmed Hassan',
        role: 'Team Lead',
        email: 'ahmed.hassan@vocalmind.io',
        activeCalls: 2,
        avgScore: 4.6,
        status: 'active',
        avatar: 'AH',
    },
    {
        id: 3,
        name: 'Michael Chen',
        role: 'Senior Agent',
        email: 'michael.chen@vocalmind.io',
        activeCalls: 5,
        avgScore: 4.4,
        status: 'active',
        avatar: 'MC',
    },
    {
        id: 4,
        name: 'Fatima Al-Rashid',
        role: 'Senior Agent',
        email: 'fatima.rashid@vocalmind.io',
        activeCalls: 3,
        avgScore: 4.2,
        status: 'active',
        avatar: 'FA',
    },
    {
        id: 5,
        name: 'James Wilson',
        role: 'Agent',
        email: 'james.wilson@vocalmind.io',
        activeCalls: 8,
        avgScore: 3.9,
        status: 'active',
        avatar: 'JW',
    },
    {
        id: 6,
        name: 'Layla Mahmoud',
        role: 'Agent',
        email: 'layla.mahmoud@vocalmind.io',
        activeCalls: 12,
        avgScore: 3.7,
        status: 'active',
        avatar: 'LM',
    },
    {
        id: 7,
        name: 'David Martinez',
        role: 'Agent',
        email: 'david.martinez@vocalmind.io',
        activeCalls: 0,
        avgScore: 3.5,
        status: 'offline',
        avatar: 'DM',
    },
    {
        id: 8,
        name: 'Nour Ibrahim',
        role: 'Agent',
        email: 'nour.ibrahim@vocalmind.io',
        activeCalls: 6,
        avgScore: 3.2,
        status: 'active',
        avatar: 'NI',
    },
    {
        id: 9,
        name: 'Emily Thompson',
        role: 'Agent',
        email: 'emily.thompson@vocalmind.io',
        activeCalls: 0,
        avgScore: 2.8,
        status: 'offline',
        avatar: 'ET',
    },
    {
        id: 10,
        name: 'Omar Khalil',
        role: 'Agent',
        email: 'omar.khalil@vocalmind.io',
        activeCalls: 0,
        avgScore: 2.1,
        status: 'offline',
        avatar: 'OK',
    },
]

// Role badge colors
const getRoleBadgeStyle = (role) => {
    switch (role) {
        case 'Manager':
            return 'bg-purple-100 text-purple-700 border-purple-200'
        case 'Team Lead':
            return 'bg-cyan/20 text-cyan-dark border-cyan/30'
        case 'Senior Agent':
            return 'bg-blue-100 text-blue-700 border-blue-200'
        default:
            return 'bg-gray-100 text-gray-700 border-gray-200'
    }
}

export default function TeamPage() {
    const router = useRouter()
    const [roleFilter, setRoleFilter] = useState('All')
    const [statusFilter, setStatusFilter] = useState('All')

    // Calculate team statistics
    const totalAgents = mockTeamMembers.length
    const activeNow = mockTeamMembers.filter((m) => m.status === 'active').length
    const avgTeamScore = (
        mockTeamMembers.reduce((acc, m) => acc + m.avgScore, 0) / totalAgents
    ).toFixed(1)
    const totalActiveCalls = mockTeamMembers.reduce((acc, m) => acc + m.activeCalls, 0)

    // Filter team members
    const filteredMembers = mockTeamMembers.filter((member) => {
        const roleMatch =
            roleFilter === 'All' ||
            member.role === roleFilter ||
            (roleFilter === 'Agent' && (member.role === 'Agent' || member.role === 'Senior Agent'))
        const statusMatch =
            statusFilter === 'All' ||
            (statusFilter === 'Active' && member.status === 'active') ||
            (statusFilter === 'Inactive' && member.status === 'offline')
        return roleMatch && statusMatch
    })

    return (
        <div className="max-w-full">
            <div className="flex flex-col lg:flex-row gap-6">
                {/* Main Content */}
                <div className="flex-1 w-full min-w-0">
                    <PageHeader title="Team Members" subtitle="Manage your customer service team">
                        <button className="px-4 py-2 bg-cyan text-navy font-medium rounded-lg hover:bg-cyan-light transition-all duration-200 flex items-center gap-2 shadow-sm whitespace-nowrap">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                                />
                            </svg>
                            Add Member
                        </button>
                    </PageHeader>

                    {/* Filters */}
                    <div className="flex flex-col md:flex-row items-start md:items-center gap-3 mb-6 overflow-x-auto pb-2 md:pb-0">
                        <div className="relative">
                            <select
                                value={roleFilter}
                                onChange={(e) => setRoleFilter(e.target.value)}
                                className="appearance-none px-4 py-2 pr-10 bg-white border border-gray-200 rounded-lg hover:border-cyan transition-all duration-200 text-gray-700 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan/50"
                            >
                                <option value="All">All Roles</option>
                                <option value="Agent">Agent</option>
                                <option value="Senior Agent">Senior Agent</option>
                                <option value="Team Lead">Team Lead</option>
                                <option value="Manager">Manager</option>
                            </select>
                            <svg
                                className="w-4 h-4 text-gray-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M19 9l-7 7-7-7"
                                />
                            </svg>
                        </div>

                        <div className="relative">
                            <select
                                value={statusFilter}
                                onChange={(e) => setStatusFilter(e.target.value)}
                                className="appearance-none px-4 py-2 pr-10 bg-white border border-gray-200 rounded-lg hover:border-cyan transition-all duration-200 text-gray-700 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan/50"
                            >
                                <option value="All">All Status</option>
                                <option value="Active">Active</option>
                                <option value="Inactive">Inactive</option>
                            </select>
                            <svg
                                className="w-4 h-4 text-gray-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M19 9l-7 7-7-7"
                                />
                            </svg>
                        </div>

                        {(roleFilter !== 'All' || statusFilter !== 'All') && (
                            <button
                                onClick={() => {
                                    setRoleFilter('All')
                                    setStatusFilter('All')
                                }}
                                className="text-cyan hover:text-cyan-dark transition-colors text-sm whitespace-nowrap"
                            >
                                Clear all
                            </button>
                        )}
                    </div>

                    {/* Team Table */}
                    <Table
                        data={filteredMembers}
                        pagination={{
                            currentPage: 1,
                            totalPages: 1,
                            totalRecords: totalAgents,
                            recordsPerPage: 10,
                            onPageChange: () => { },
                        }}
                        columns={[
                            {
                                header: 'NAME',
                                cell: (member) => (
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan to-purple flex items-center justify-center text-white text-sm font-medium">
                                            {member.avatar}
                                        </div>
                                        <span
                                            className="font-medium text-gray-900 hover:text-cyan cursor-pointer"
                                            onClick={() => router.push(`/manager/agent/${member.id}`)}
                                        >
                                            {member.name}
                                        </span>
                                    </div>
                                ),
                            },
                            {
                                header: 'ROLE',
                                cell: (member) => (
                                    <span
                                        className={`px-2.5 py-1 text-xs font-medium rounded-full border ${getRoleBadgeStyle(member.role)}`}
                                    >
                                        {member.role}
                                    </span>
                                ),
                            },
                            {
                                header: 'EMAIL',
                                accessor: 'email',
                                className: 'text-gray-600 hidden md:table-cell',
                                headerClassName: 'hidden md:table-cell',
                            },
                            {
                                header: 'ACTIVE CALLS',
                                headerClassName: 'text-center hidden md:table-cell',
                                className: 'text-center hidden md:table-cell',
                                cell: (member) => (
                                    <span
                                        className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${member.activeCalls > 0
                                            ? 'bg-cyan/20 text-cyan-dark'
                                            : 'bg-gray-100 text-gray-500'
                                            }`}
                                    >
                                        {member.activeCalls}
                                    </span>
                                ),
                            },
                            {
                                header: 'PERFORMANCE',
                                headerClassName: 'hidden md:table-cell',
                                className: 'hidden md:table-cell',
                                cell: (member) => <StarRating score={member.avgScore} />,
                            },
                            {
                                header: 'STATUS',
                                cell: (member) => <StatusBadge status={member.status} showDot />,
                            },
                            {
                                header: 'ACTIONS',
                                headerClassName: 'hidden md:table-cell',
                                className: 'hidden md:table-cell',
                                cell: (member) => (
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => router.push(`/manager/agent/${member.id}`)}
                                            className="p-2 text-gray-400 hover:text-cyan hover:bg-cyan/10 rounded-lg transition-all duration-200"
                                            title="View Profile"
                                        >
                                            <svg
                                                className="w-4 h-4"
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                            >
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                                />
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                                />
                                            </svg>
                                        </button>
                                        <button
                                            className="p-2 text-gray-400 hover:text-amber-500 hover:bg-amber-50 rounded-lg transition-all duration-200"
                                            title="Edit"
                                        >
                                            <svg
                                                className="w-4 h-4"
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                            >
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                                                />
                                            </svg>
                                        </button>
                                        <button
                                            className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all duration-200"
                                            title="Deactivate"
                                        >
                                            <svg
                                                className="w-4 h-4"
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                            >
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"
                                                />
                                            </svg>
                                        </button>
                                    </div>
                                ),
                            },
                        ]}
                    />
                </div>

                {/* Stats Sidebar */}
                <div className="w-full lg:w-72 flex-shrink-0">
                    <div className="bg-white rounded-xl shadow-card p-6 sticky top-6">
                        <h2 className="text-lg font-semibold text-gray-900 mb-6">Team Overview</h2>

                        {/* Total Agents */}
                        <div className="mb-6 p-4 bg-gradient-to-br from-navy to-navy-light rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-gray-300">Total Agents</span>
                                <svg
                                    className="w-5 h-5 text-cyan"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                                    />
                                </svg>
                            </div>
                            <p className="text-3xl font-bold text-white">{totalAgents}</p>
                        </div>

                        {/* Active Now */}
                        <div className="mb-6 p-4 bg-emerald-50 rounded-lg border border-emerald-100">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-emerald-600">Active Now</span>
                                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                            </div>
                            <p className="text-3xl font-bold text-emerald-700">{activeNow}</p>
                            <p className="text-xs text-emerald-600 mt-1">{totalActiveCalls} active calls</p>
                        </div>

                        {/* Avg Team Score */}
                        <div className="p-4 bg-gradient-to-br from-cyan/10 to-purple/10 rounded-lg border border-cyan/20">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-gray-600">Avg Team Score</span>
                                <svg className="w-5 h-5 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                </svg>
                            </div>
                            <p className="text-3xl font-bold text-gray-900">{avgTeamScore}</p>
                            <div className="flex items-center gap-1 mt-2">
                                <StarRating score={parseFloat(avgTeamScore)} />
                            </div>
                        </div>

                        {/* Quick Actions */}
                        <div className="mt-6 pt-6 border-t border-gray-100">
                            <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Actions</h3>
                            <div className="space-y-2">
                                <button className="w-full px-4 py-2 text-sm text-left text-gray-600 hover:bg-gray-50 rounded-lg transition-colors flex items-center gap-3">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                        />
                                    </svg>
                                    Generate Report
                                </button>
                                <button className="w-full px-4 py-2 text-sm text-left text-gray-600 hover:bg-gray-50 rounded-lg transition-colors flex items-center gap-3">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                                        />
                                    </svg>
                                    Schedule Training
                                </button>
                                <button className="w-full px-4 py-2 text-sm text-left text-gray-600 hover:bg-gray-50 rounded-lg transition-colors flex items-center gap-3">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                                        />
                                    </svg>
                                    Manage Shifts
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
