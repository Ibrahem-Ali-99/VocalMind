import React from 'react'
import LoadingSpinner from './LoadingSpinner'
import EmptyState from './EmptyState'

const Table = ({
    columns,
    data,
    onRowClick,
    pagination,
    isLoading = false,
    emptyMessage = 'No data found',
    minWidth = '1000px',
}) => {
    if (isLoading) {
        return (
            <div className="bg-white rounded-xl shadow-card p-12 flex justify-center items-center h-64">
                <LoadingSpinner />
            </div>
        )
    }

    if (!data || data.length === 0) {
        return (
            <div className="bg-white rounded-xl shadow-card">
                <EmptyState
                    title={emptyMessage}
                    description="Try adjusting your filters." // Could also be a prop
                />
            </div>
        )
    }

    return (
        <div className="bg-white rounded-xl shadow-card overflow-hidden border border-gray-100">
            <div className="overflow-x-auto">
                <table className="w-full" style={{ minWidth }}>
                    <thead>
                        <tr className="text-left text-xs font-semibold text-gray-500 border-b border-gray-100 bg-gray-50/50 uppercase tracking-wider">
                            {columns.map((col, idx) => (
                                <th key={idx} className={`px-6 py-4 ${col.headerClassName || ''}`}>
                                    {col.header}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                        {data.map((row, rowIdx) => (
                            <tr
                                key={row.id || rowIdx}
                                className={`transition-colors ${onRowClick ? 'cursor-pointer hover:bg-cyan/5' : ''}`}
                                onClick={() => onRowClick && onRowClick(row)}
                            >
                                {columns.map((col, colIdx) => (
                                    <td key={colIdx} className={`px-6 py-4 ${col.className || ''}`}>
                                        {col.cell
                                            ? col.cell(row)
                                            : col.accessor && typeof col.accessor === 'function'
                                                ? col.accessor(row)
                                                : row[col.accessor]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Pagination */}
            {pagination && pagination.totalPages > 1 && (
                <div className="px-6 py-4 border-t border-gray-100 flex flex-col sm:flex-row items-center justify-between gap-4 bg-gray-50/30">
                    <p className="text-sm text-gray-500 text-center sm:text-left">
                        Showing{' '}
                        <span className="font-medium">
                            {(pagination.currentPage - 1) * pagination.recordsPerPage + 1}
                        </span>{' '}
                        to{' '}
                        <span className="font-medium">
                            {Math.min(
                                pagination.currentPage * pagination.recordsPerPage,
                                pagination.totalRecords
                            )}
                        </span>{' '}
                        of <span className="font-medium">{pagination.totalRecords}</span> results
                    </p>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => pagination.onPageChange(Math.max(1, pagination.currentPage - 1))}
                            disabled={pagination.currentPage === 1}
                            className="px-3 py-1.5 text-sm font-medium border border-gray-200 rounded-lg hover:bg-white hover:border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            Previous
                        </button>
                        <button
                            onClick={() =>
                                pagination.onPageChange(Math.min(pagination.totalPages, pagination.currentPage + 1))
                            }
                            disabled={pagination.currentPage === pagination.totalPages}
                            className="px-3 py-1.5 text-sm font-medium border border-gray-200 rounded-lg hover:bg-white hover:border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            Next
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default Table
