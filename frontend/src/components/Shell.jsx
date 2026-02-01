'use client'

import { useState } from 'react'
import { usePathname } from 'next/navigation'
import Header from '@/components/shared/Header'
import Sidebar from '@/components/shared/Sidebar'

// Routes that should not show the shell (sidebar/header)
const noShellRoutes = ['/login']

export default function Shell({ children }) {
    const pathname = usePathname()
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
    const [mobileOpen, setMobileOpen] = useState(false)

    // Agent Navigation Items
    const agentNavItems = [
        { name: 'Dashboard', path: '/agent/dashboard', icon: 'Dashboard' },
        { name: 'Calls', path: '/agent/calls', icon: 'Calls' },
        { name: 'Performance', path: '/agent/performance', icon: 'Reports' },
        { name: 'Training', path: '/agent/training', icon: 'Policies' },
    ]

    // Don't render shell for login and other auth pages
    if (noShellRoutes.some(route => pathname?.startsWith(route))) {
        return <>{children}</>
    }

    const isAgent = pathname?.startsWith('/agent')
    const navItems = isAgent ? agentNavItems : undefined

    const handleHeaderToggle = () => {
        if (typeof window !== 'undefined' && window.innerWidth >= 768) {
            setSidebarCollapsed(!sidebarCollapsed)
        } else {
            setMobileOpen(!mobileOpen)
        }
    }

    return (
        <div className="flex min-h-screen bg-gray-50">
            <Sidebar
                isCollapsed={sidebarCollapsed}
                mobileOpen={mobileOpen}
                onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
                navItems={navItems}
            />

            {/* Mobile overlay */}
            {mobileOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-20 md:hidden"
                    onClick={() => setMobileOpen(false)}
                />
            )}

            <div className="flex-1 flex flex-col min-w-0">
                <Header onMenuClick={handleHeaderToggle} />
                <main className="flex-1 p-4 md:p-6 overflow-auto">
                    {children}
                </main>
            </div>
        </div>
    )
}
