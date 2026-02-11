/**
 * API Client with mock switcher
 */
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api'
const USE_MOCKS = process.env.NEXT_PUBLIC_USE_MOCKS === 'true'

export async function fetchSession(sessionId) {
    if (USE_MOCKS) {
        // Return mock data
        const { mockSessionData } = await import('./mocks/sessionData')
        return mockSessionData
    }
    const res = await fetch(`${API_BASE}/sessions/${sessionId}`)
    if (!res.ok) throw new Error('Failed to fetch session')
    return res.json()
}

export async function fetchCalls(filters = {}) {
    if (USE_MOCKS) {
        const { mockCallsData } = await import('./mocks/callsData')
        return mockCallsData
    }
    const res = await fetch(`${API_BASE}/calls`)
    if (!res.ok) throw new Error('Failed to fetch calls')
    return res.json()
}

export async function fetchDashboardStats() {
    if (USE_MOCKS) {
        const { mockDashboardStats } = await import('./mocks/dashboardData')
        return mockDashboardStats
    }
    const res = await fetch(`${API_BASE}/dashboard/stats`)
    if (!res.ok) throw new Error('Failed to fetch dashboard stats')
    return res.json()
}
