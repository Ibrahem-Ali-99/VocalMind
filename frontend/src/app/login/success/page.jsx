'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuth } from '@/components/providers/AuthProvider'

export default function LoginSuccessPage() {
    const { login } = useAuth()
    const router = useRouter()
    const searchParams = useSearchParams()

    useEffect(() => {
        const token = searchParams.get('token')
        if (token) {
            login(token)
        } else {
            router.push('/login')
        }
    }, [router, searchParams, login])

    return (
        <div className="min-h-screen flex items-center justify-center bg-navy text-white">
            <div className="flex flex-col items-center gap-4">
                <div className="animate-spin h-8 w-8 border-4 border-cyan rounded-full border-t-transparent"></div>
                <p>Authenticating...</p>
            </div>
        </div>
    )
}
