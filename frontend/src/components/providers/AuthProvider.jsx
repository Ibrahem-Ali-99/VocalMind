'use client'

import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { useRouter, usePathname } from 'next/navigation'

const AuthContext = createContext({
    user: null,
    login: () => { },
    logout: () => { },
    loading: true,
})

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)
    const router = useRouter()
    const pathname = usePathname()

    useEffect(() => {
        // Check for token in localStorage on mount
        const token = localStorage.getItem('token')
        if (token) {
            try {
                // Parse JWT payload (part 2 of the token)
                const payload = JSON.parse(atob(token.split('.')[1]))

                // Check if token is expired
                const isExpired = payload.exp * 1000 < Date.now()
                if (isExpired) {
                    localStorage.removeItem('token')
                    setUser(null)
                } else {
                    setUser({
                        id: payload.sub, // 'sub' is the user ID from backend
                        email: payload.sub, // we can fetch real profile later
                    })
                }
            } catch (e) {
                console.error("Invalid token format", e)
                localStorage.removeItem('token')
            }
        }
        setLoading(false)
    }, [])

    const login = useCallback((token) => {
        localStorage.setItem('token', token)
        try {
            const payload = JSON.parse(atob(token.split('.')[1]))
            setUser({ id: payload.sub })
            router.push('/manager/dashboard')
        } catch (e) {
            console.error("Login failed: Invalid token", e)
        }
    }, [router])

    const logout = useCallback(() => {
        localStorage.removeItem('token')
        setUser(null)
        router.push('/login')
    }, [router])

    // Client-side Route Protection
    useEffect(() => {
        const publicPaths = ['/login', '/login/success', '/']
        // specific check for public path startsWith to allow sub-paths if needed, but simplistic here
        const isPublic = publicPaths.some(path => pathname === path || pathname?.startsWith('/login'))

        if (!loading && !user && !isPublic) {
            router.push('/login')
        }
    }, [user, loading, pathname, router])

    return (
        <AuthContext.Provider value={{ user, login, logout, loading }}>
            {children}
        </AuthContext.Provider>
    )
}

export const useAuth = () => useContext(AuthContext)
