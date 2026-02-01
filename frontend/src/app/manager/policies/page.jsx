'use client'

import { useState } from 'react'

// Mock policies data
const mockPolicies = [
    {
        id: 1,
        title: 'Customer Service Guidelines',
        category: 'Customer Service',
        uploadDate: 'Jan 28, 2026',
        status: 'Active',
        version: '2.3',
        embeddingStatus: 'Processed',
        description: 'Standard operating procedures for customer interactions',
        versions: [
            { version: '2.3', date: 'Jan 28, 2026', author: 'Sarah Miller' },
            { version: '2.2', date: 'Dec 15, 2025', author: 'Sarah Miller' },
            { version: '2.1', date: 'Oct 10, 2025', author: 'John Smith' },
        ],
    },
    {
        id: 2,
        title: 'Data Privacy & Security Policy',
        category: 'Compliance',
        uploadDate: 'Jan 25, 2026',
        status: 'Active',
        version: '1.5',
        embeddingStatus: 'Processed',
        description: 'GDPR and data protection compliance requirements',
        versions: [
            { version: '1.5', date: 'Jan 25, 2026', author: 'Legal Team' },
            { version: '1.4', date: 'Nov 20, 2025', author: 'Legal Team' },
        ],
    },
    {
        id: 3,
        title: 'Product Knowledge Base',
        category: 'Product Info',
        uploadDate: 'Jan 20, 2026',
        status: 'Active',
        version: '4.1',
        embeddingStatus: 'Processing',
        description: 'Complete product documentation and FAQs',
        versions: [{ version: '4.1', date: 'Jan 20, 2026', author: 'Product Team' }],
    },
    {
        id: 4,
        title: 'Escalation Procedures',
        category: 'Customer Service',
        uploadDate: 'Jan 15, 2026',
        status: 'Active',
        version: '1.2',
        embeddingStatus: 'Processed',
        description: 'When and how to escalate customer issues',
        versions: [
            { version: '1.2', date: 'Jan 15, 2026', author: 'Sarah Miller' },
            { version: '1.1', date: 'Dec 1, 2025', author: 'Sarah Miller' },
        ],
    },
    {
        id: 5,
        title: 'Refund Policy Guidelines',
        category: 'Compliance',
        uploadDate: 'Jan 10, 2026',
        status: 'Draft',
        version: '0.9',
        embeddingStatus: 'Failed',
        description: 'Refund eligibility and process documentation',
        versions: [{ version: '0.9', date: 'Jan 10, 2026', author: 'Finance Team' }],
    },
    {
        id: 6,
        title: 'New Agent Onboarding Guide',
        category: 'Training',
        uploadDate: 'Dec 28, 2025',
        status: 'Active',
        version: '3.0',
        embeddingStatus: 'Processed',
        description: 'Complete training materials for new team members',
        versions: [{ version: '3.0', date: 'Dec 28, 2025', author: 'HR Team' }],
    },
]

const categories = ['All', 'Customer Service', 'Compliance', 'Product Info', 'Training']

// Status badge
const StatusBadge = ({ status }) => {
    const styles = {
        Active: 'bg-emerald-100 text-emerald-700',
        Draft: 'bg-amber-100 text-amber-700',
        Archived: 'bg-gray-100 text-gray-600',
    }
    return (
        <span
            className={`px-2 py-0.5 text-xs font-medium rounded ${styles[status] || styles['Draft']}`}
        >
            {status}
        </span>
    )
}

// Embedding status indicator
const EmbeddingStatus = ({ status }) => {
    const config = {
        Processed: { color: 'text-emerald-500', bg: 'bg-emerald-100', icon: '✓' },
        Processing: { color: 'text-cyan-500', bg: 'bg-cyan/20', icon: '⟳' },
        Failed: { color: 'text-red-500', bg: 'bg-red-100', icon: '✕' },
    }
    const { color, bg, icon } = config[status] || config['Processing']
    return (
        <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-xs ${bg}`}>
            <span className={color}>{icon}</span>
            <span className={`${color} font-medium`}>{status}</span>
        </div>
    )
}

// Policy Card Component
const PolicyCard = ({ policy, onPreview, onEdit, onDelete, onViewHistory }) => (
    <div className="bg-white rounded-xl shadow-card p-5 hover:shadow-card-hover transition-all duration-200">
        <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple/20 to-cyan/20 flex items-center justify-center">
                    <svg
                        className="w-5 h-5 text-purple"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                        />
                    </svg>
                </div>
                <div>
                    <h3 className="font-medium text-gray-900">{policy.title}</h3>
                    <p className="text-xs text-gray-500">{policy.category}</p>
                </div>
            </div>
            <StatusBadge status={policy.status} />
        </div>

        <p className="text-sm text-gray-600 mb-4">{policy.description}</p>

        <div className="flex items-center justify-between text-xs text-gray-400 mb-4">
            <span>Uploaded: {policy.uploadDate}</span>
            <span>v{policy.version}</span>
        </div>

        <div className="flex items-center justify-between pt-4 border-t border-gray-100">
            <EmbeddingStatus status={policy.embeddingStatus} />
            <div className="flex items-center gap-2">
                <button
                    onClick={() => onViewHistory(policy)}
                    className="p-2 text-gray-400 hover:text-purple hover:bg-purple/10 rounded-lg transition-all"
                    title="Version History"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                    </svg>
                </button>
                <button
                    onClick={() => onPreview(policy)}
                    className="p-2 text-gray-400 hover:text-cyan hover:bg-cyan/10 rounded-lg transition-all"
                    title="Preview"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                    onClick={() => onEdit(policy)}
                    className="p-2 text-gray-400 hover:text-amber-500 hover:bg-amber-50 rounded-lg transition-all"
                    title="Edit"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                        />
                    </svg>
                </button>
                <button
                    onClick={() => onDelete(policy)}
                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                    title="Delete"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                    </svg>
                </button>
            </div>
        </div>
    </div>
)

// Version History Modal
const VersionHistoryModal = ({ policy, onClose }) => (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Version History</h3>
                <button onClick={onClose} className="p-1 text-gray-400 hover:text-gray-600">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                        />
                    </svg>
                </button>
            </div>
            <p className="text-sm text-gray-500 mb-4">{policy.title}</p>
            <div className="space-y-3">
                {policy.versions.map((v, i) => (
                    <div
                        key={i}
                        className={`p-3 rounded-lg ${i === 0 ? 'bg-cyan/10 border border-cyan/30' : 'bg-gray-50'}`}
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="font-medium text-gray-900">v{v.version}</span>
                                {i === 0 && (
                                    <span className="px-1.5 py-0.5 text-xs bg-cyan text-navy rounded">Current</span>
                                )}
                            </div>
                            <span className="text-xs text-gray-400">{v.date}</span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">By {v.author}</p>
                    </div>
                ))}
            </div>
        </div>
    </div>
)

export default function PoliciesPage() {
    const [searchQuery, setSearchQuery] = useState('')
    const [categoryFilter, setCategoryFilter] = useState('All')
    const [viewMode, setViewMode] = useState('grid') // 'grid' or 'list'
    const [selectedPolicy, setSelectedPolicy] = useState(null)
    const [showVersionHistory, setShowVersionHistory] = useState(false)

    const [statusFilter, setStatusFilter] = useState('All') // 'All', 'Active', 'Processed', 'Failed'

    // Base filter (Search + Category)
    const baseFilteredPolicies = mockPolicies.filter((policy) => {
        const matchesSearch =
            policy.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            policy.description.toLowerCase().includes(searchQuery.toLowerCase())
        const matchesCategory = categoryFilter === 'All' || policy.category === categoryFilter
        return matchesSearch && matchesCategory
    })

    // Apply status filter for display
    const displayedPolicies = baseFilteredPolicies.filter((policy) => {
        if (statusFilter === 'All') return true
        if (statusFilter === 'Active') return policy.status === 'Active'
        if (statusFilter === 'Processed') return policy.embeddingStatus === 'Processed'
        if (statusFilter === 'Failed') return policy.embeddingStatus === 'Failed'
        return true
    })

    const handlePreview = () => {
        // Would open preview modal
    }

    const handleEdit = () => {
        // Would open edit modal
    }

    const handleDelete = () => {
        // Would show confirmation dialog
    }

    const handleViewHistory = (policy) => {
        setSelectedPolicy(policy)
        setShowVersionHistory(true)
    }

    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-900">Company Policies</h1>
                    <p className="text-gray-500 mt-1">Manage policy documents for AI compliance checking</p>
                </div>
                <button className="px-4 py-2 bg-cyan text-navy font-medium rounded-lg hover:bg-cyan-light transition-all flex items-center gap-2 w-full sm:w-auto justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                    </svg>
                    Upload New Policy
                </button>
            </div>

            {/* Filters & Search */}
            <div className="bg-white rounded-xl shadow-card p-4 mb-6">
                <div className="flex flex-col lg:flex-row lg:items-center gap-4">
                    {/* Search */}
                    <div className="flex-1 relative">
                        <svg
                            className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                            />
                        </svg>
                        <input
                            type="text"
                            placeholder="Search policies..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan/50"
                        />
                    </div>

                    {/* Category Filter */}
                    <div className="flex items-center gap-2 overflow-x-auto pb-2 lg:pb-0 no-scrollbar">
                        {categories.map((cat) => (
                            <button
                                key={cat}
                                onClick={() => setCategoryFilter(cat)}
                                className={`px-3 py-1.5 text-sm rounded-lg transition-all whitespace-nowrap ${categoryFilter === cat
                                    ? 'bg-cyan text-navy font-medium'
                                    : 'text-gray-600 hover:bg-gray-100'
                                    }`}
                            >
                                {cat}
                            </button>
                        ))}
                    </div>

                    {/* View Toggle */}
                    <div className="flex items-center gap-1 p-1 bg-gray-100 rounded-lg w-fit ml-auto lg:ml-0">
                        <button
                            onClick={() => setViewMode('grid')}
                            className={`p-2 rounded ${viewMode === 'grid' ? 'bg-white shadow' : 'text-gray-400 hover:text-gray-600'}`}
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
                                />
                            </svg>
                        </button>
                        <button
                            onClick={() => setViewMode('list')}
                            className={`p-2 rounded ${viewMode === 'list' ? 'bg-white shadow' : 'text-gray-400 hover:text-gray-600'}`}
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 6h16M4 12h16M4 18h16"
                                />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div
                    onClick={() => setStatusFilter('All')}
                    className={`bg-white rounded-lg p-4 shadow-card cursor-pointer transition-all border-2 ${statusFilter === 'All' ? 'border-cyan' : 'border-transparent hover:border-gray-200'}`}
                >
                    <p className="text-sm text-gray-500">Total Policies</p>
                    <p className="text-2xl font-bold text-gray-900">{baseFilteredPolicies.length}</p>
                </div>
                <div
                    onClick={() => setStatusFilter('Active')}
                    className={`bg-white rounded-lg p-4 shadow-card cursor-pointer transition-all border-2 ${statusFilter === 'Active' ? 'border-emerald-500' : 'border-transparent hover:border-emerald-200'}`}
                >
                    <p className="text-sm text-gray-500">Active</p>
                    <p className="text-2xl font-bold text-emerald-600">
                        {baseFilteredPolicies.filter((p) => p.status === 'Active').length}
                    </p>
                </div>
                <div
                    onClick={() => setStatusFilter('Processed')}
                    className={`bg-white rounded-lg p-4 shadow-card cursor-pointer transition-all border-2 ${statusFilter === 'Processed' ? 'border-cyan' : 'border-transparent hover:border-cyan/30'}`}
                >
                    <p className="text-sm text-gray-500">Processed</p>
                    <p className="text-2xl font-bold text-cyan-600">
                        {baseFilteredPolicies.filter((p) => p.embeddingStatus === 'Processed').length}
                    </p>
                </div>
                <div
                    onClick={() => setStatusFilter('Failed')}
                    className={`bg-white rounded-lg p-4 shadow-card cursor-pointer transition-all border-2 ${statusFilter === 'Failed' ? 'border-red-500' : 'border-transparent hover:border-red-200'}`}
                >
                    <p className="text-sm text-gray-500">Needs Attention</p>
                    <p className="text-2xl font-bold text-red-500">
                        {baseFilteredPolicies.filter((p) => p.embeddingStatus === 'Failed').length}
                    </p>
                </div>
            </div>

            {/* Policies Grid/List */}
            {viewMode === 'grid' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {displayedPolicies.map((policy) => (
                        <PolicyCard
                            key={policy.id}
                            policy={policy}
                            onPreview={handlePreview}
                            onEdit={handleEdit}
                            onDelete={handleDelete}
                            onViewHistory={handleViewHistory}
                        />
                    ))}
                </div>
            ) : (
                <div className="bg-white rounded-xl shadow-card overflow-hidden overflow-x-auto">
                    <table className="w-full min-w-[800px]">
                        <thead>
                            <tr className="text-left text-sm text-gray-500 border-b border-gray-100 bg-gray-50">
                                <th className="px-6 py-3 font-medium">Title</th>
                                <th className="px-6 py-3 font-medium">Category</th>
                                <th className="px-6 py-3 font-medium">Version</th>
                                <th className="px-6 py-3 font-medium">Upload Date</th>
                                <th className="px-6 py-3 font-medium">Status</th>
                                <th className="px-6 py-3 font-medium">Embedding</th>
                                <th className="px-6 py-3 font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {displayedPolicies.map((policy) => (
                                <tr key={policy.id} className="border-b border-gray-50 hover:bg-gray-50">
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded bg-purple/10 flex items-center justify-center">
                                                <svg
                                                    className="w-4 h-4 text-purple"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    viewBox="0 0 24 24"
                                                >
                                                    <path
                                                        strokeLinecap="round"
                                                        strokeLinejoin="round"
                                                        strokeWidth={2}
                                                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                                    />
                                                </svg>
                                            </div>
                                            <span className="font-medium text-gray-900">{policy.title}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-gray-600 text-sm">{policy.category}</td>
                                    <td className="px-6 py-4 text-gray-600 text-sm">v{policy.version}</td>
                                    <td className="px-6 py-4 text-gray-500 text-sm">{policy.uploadDate}</td>
                                    <td className="px-6 py-4">
                                        <StatusBadge status={policy.status} />
                                    </td>
                                    <td className="px-6 py-4">
                                        <EmbeddingStatus status={policy.embeddingStatus} />
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-1">
                                            <button
                                                onClick={() => handleViewHistory(policy)}
                                                className="p-1.5 text-gray-400 hover:text-purple"
                                                title="History"
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
                                                        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                                                    />
                                                </svg>
                                            </button>
                                            <button
                                                onClick={() => handlePreview(policy)}
                                                className="p-1.5 text-gray-400 hover:text-cyan"
                                                title="Preview"
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
                                                </svg>
                                            </button>
                                            <button
                                                onClick={() => handleEdit(policy)}
                                                className="p-1.5 text-gray-400 hover:text-amber-500"
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
                                                onClick={() => handleDelete(policy)}
                                                className="p-1.5 text-gray-400 hover:text-red-500"
                                                title="Delete"
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
                                                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                                                    />
                                                </svg>
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {displayedPolicies.length === 0 && (
                <div className="bg-white rounded-xl shadow-card p-12 text-center">
                    <svg
                        className="w-12 h-12 text-gray-300 mx-auto mb-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                        />
                    </svg>
                    <p className="text-gray-500">No policies found matching your criteria</p>
                </div>
            )}

            {/* Version History Modal */}
            {showVersionHistory && selectedPolicy && (
                <VersionHistoryModal
                    policy={selectedPolicy}
                    onClose={() => {
                        setShowVersionHistory(false)
                        setSelectedPolicy(null)
                    }}
                />
            )}
        </div>
    )
}
