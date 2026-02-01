import React from 'react'

const PageHeader = ({ title, subtitle, children }) => {
    return (
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
            <div>
                <h1 className="text-2xl font-semibold text-gray-900">{title}</h1>
                {subtitle && <p className="text-gray-500 mt-1">{subtitle}</p>}
            </div>
            {children && (
                <div className="flex flex-col sm:flex-row items-center gap-3 w-full md:w-auto">
                    {children}
                </div>
            )}
        </div>
    )
}

export default PageHeader
