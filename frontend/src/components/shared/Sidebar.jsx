'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'

// Icon components
const DashboardIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
        />
    </svg>
)

const CallsIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
        />
    </svg>
)

const TeamIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"
        />
    </svg>
)

const ReportsIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
    </svg>
)

const SettingsIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
        />
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
    </svg>
)

const UploadIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
    </svg>
)

const PoliciesIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
    </svg>
)

const NavIcon = ({ name }) => {
    switch (name) {
        case 'Dashboard':
            return <DashboardIcon />
        case 'Upload':
            return <UploadIcon />
        case 'Calls':
            return <CallsIcon />
        case 'Team':
            return <TeamIcon />
        case 'Reports':
            return <ReportsIcon />
        case 'Policies':
            return <PoliciesIcon />
        case 'Settings':
            return <SettingsIcon />
        default:
            return null
    }
}

export default function Sidebar({ isCollapsed, mobileOpen, onToggle, onClose, navItems }) {
    const pathname = usePathname()

    const items = navItems || [
        { name: 'Dashboard', path: '/manager/dashboard', icon: 'Dashboard' },
        { name: 'Upload', path: '/manager/upload', icon: 'Upload' },
        { name: 'Calls', path: '/manager/calls', icon: 'Calls' },
        { name: 'Team', path: '/manager/team', icon: 'Team' },
        { name: 'Reports', path: '/manager/reports', icon: 'Reports' },
        { name: 'Policies', path: '/manager/policies', icon: 'Policies' },
    ]

    return (
        <aside
            className={`fixed inset-y-0 left-0 z-30 bg-navy text-white transition-all duration-300 ease-in-out
        md:relative md:translate-x-0
        ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}
        ${isCollapsed ? 'md:w-20' : 'md:w-64'}
        w-64 flex flex-col overflow-x-hidden`}
        >
            {/* Logo Area */}
            <div className="h-16 flex items-center px-6 border-b border-navy-light relative">
                <Link
                    href={items[0]?.path || '/'}
                    className="flex items-center gap-3 hover:opacity-80 transition-opacity flex-1"
                >
                    <div className={`flex items-center gap-3 ${isCollapsed ? 'justify-center w-full' : ''}`}>
                        <Image src="/images/logo-icon.svg" alt="" width={32} height={32} className="flex-shrink-0 w-8 h-8" />
                        {!isCollapsed && (
                            <span className="font-bold text-xl tracking-tight text-white whitespace-nowrap">
                                VocalMind
                            </span>
                        )}
                    </div>
                </Link>
                {/* Mobile close button */}
                {onClose && (
                    <button
                        onClick={onClose}
                        className="md:hidden absolute right-3 top-1/2 -translate-y-1/2 w-9 h-9 flex items-center justify-center rounded-lg text-gray-400 hover:text-white hover:bg-navy-light transition-colors"
                        aria-label="Close sidebar"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                )}
            </div>

            {/* Navigation */}
            <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto overflow-x-hidden">
                {items.map((item) => (
                    <Link
                        key={item.path}
                        href={item.path}
                        className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 group relative ${pathname === item.path ||
                            (item.path === '/calls' && pathname.startsWith('/session'))
                            ? 'bg-gradient-to-r from-cyan to-cyan-dark text-white shadow-lg shadow-cyan/20'
                            : 'text-gray-400 hover:text-white hover:bg-navy-light'
                            }`}
                    >
                        <span className="flex-shrink-0">
                            <NavIcon name={item.icon} />
                        </span>
                        {!isCollapsed && <span className="ml-3 font-medium truncate">{item.name}</span>}

                        {isCollapsed && (
                            <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                                {item.name}
                            </div>
                        )}
                    </Link>
                ))}
            </nav>

            {/* Settings */}
            <div className="mt-8 pt-4 border-t border-navy-light">
                <Link
                    href={pathname.startsWith('/agent') ? '/agent/settings' : '/manager/settings'}
                    className={`flex items-center px-3 py-3 rounded-lg transition-smooth duration-300 group relative ${pathname.includes('/settings')
                        ? 'bg-gradient-to-r from-cyan to-cyan-dark text-white shadow-lg shadow-cyan/20'
                        : 'text-gray-400 hover:text-white hover:bg-navy-light'
                        }`}
                    title={isCollapsed ? 'Settings' : ''}
                >
                    <span className="flex-shrink-0">
                        <NavIcon name="Settings" />
                    </span>
                    {!isCollapsed && <span>Settings</span>}
                </Link>
            </div>

            {/* Collapse Toggle */}
            <div className="p-3 border-t border-navy-light">
                <button
                    onClick={onToggle}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 text-gray-400 hover:text-white hover:bg-navy-light rounded-lg transition-smooth hover:bg-white/5 active:scale-95 cursor-pointer"
                    title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                    aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                    <svg
                        className={`w-5 h-5 transition-transform duration-300 ${isCollapsed ? 'rotate-180' : ''}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
                        />
                    </svg>
                </button>
            </div>
        </aside >
    )
}
