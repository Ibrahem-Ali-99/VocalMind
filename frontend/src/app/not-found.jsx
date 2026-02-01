import Link from 'next/link'

export default function NotFound() {
    return (
        <div className="flex h-screen flex-col items-center justify-center bg-gray-50 text-center px-4">
            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mb-6">
                <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">404 - Page Not Found</h2>
            <p className="text-gray-500 mb-8 max-w-md">The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.</p>
            <Link
                href="/manager/dashboard"
                className="px-6 py-3 bg-cyan text-navy font-semibold rounded-lg hover:bg-cyan-light transition-colors shadow-sm"
            >
                Return to Dashboard
            </Link>
        </div>
    )
}
