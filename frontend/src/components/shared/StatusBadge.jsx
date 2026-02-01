import React from 'react'

const getStatusStyles = (status, variant) => {
    const normalizedStatus = status?.toLowerCase()

    // Explicit variant override
    if (variant === 'success') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
    if (variant === 'warning') return 'bg-amber-50 text-amber-700 border-amber-200'
    if (variant === 'error') return 'bg-red-50 text-red-700 border-red-200'
    if (variant === 'neutral') return 'bg-gray-50 text-gray-700 border-gray-200'

    // Auto-map based on common status terms
    if (['active', 'resolved', 'completed', 'paid', 'success'].includes(normalizedStatus)) {
        return 'bg-emerald-50 text-emerald-700 border-emerald-200' // Success
    }
    if (['pending', 'in progress', 'warning', 'review'].includes(normalizedStatus)) {
        return 'bg-amber-50 text-amber-700 border-amber-200' // Warning
    }
    if (
        ['fractioned', 'escalated', 'error', 'failed', 'inactive', 'offline'].includes(normalizedStatus)
    ) {
        return normalizedStatus === 'offline' || normalizedStatus === 'inactive'
            ? 'bg-gray-100 text-gray-600 border-gray-200' // Neutral/Offline
            : 'bg-red-50 text-red-700 border-red-200' // Error/Escalated
    }

    return 'bg-gray-50 text-gray-600 border-gray-200' // Default neutral
}

const getDotColor = (styles) => {
    if (styles.includes('emerald')) return 'bg-emerald-500'
    if (styles.includes('amber')) return 'bg-amber-500'
    if (styles.includes('red')) return 'bg-red-500'
    return 'bg-gray-400'
}

const StatusBadge = ({ status, variant, showDot = false, className = '' }) => {
    const styles = getStatusStyles(status, variant)
    const dotColor = getDotColor(styles)
    const label = status || (variant ? variant.charAt(0).toUpperCase() + variant.slice(1) : 'Unknown')

    return (
        <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border ${styles} ${className}`}
        >
            {showDot && (
                <span
                    className={`w-2 h-2 rounded-full ${dotColor} ${styles.includes('emerald') ? 'animate-pulse' : ''}`}
                ></span>
            )}
            {label}
        </span>
    )
}

export default StatusBadge
