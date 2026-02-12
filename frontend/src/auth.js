import NextAuth from 'next-auth'
import Credentials from 'next-auth/providers/credentials'
import Google from 'next-auth/providers/google'

export const { handlers, signIn, signOut, auth } = NextAuth({
    secret: process.env.AUTH_SECRET || 'dev-secret-change-me-in-production',
    providers: [
        Credentials({
            name: 'Credentials',
            credentials: {
                email: { label: 'Email', type: 'email' },
                password: { label: 'Password', type: 'password' },
            },
            async authorize(credentials) {
                // Mock authentication - accept any credentials in demo mode
                if (process.env.NEXT_PUBLIC_USE_MOCKS === 'true') {
                    return {
                        id: '1',
                        name: 'Sarah Miller',
                        email: credentials?.email || 'sarah.miller@vocalmind.com',
                        role: 'manager',
                    }
                }

                // TODO: Replace with real API authentication
                // const user = await verifyCredentials(credentials.email, credentials.password)
                // if (user) return user

                return null
            },
        }),
        Google({
            clientId: process.env.GOOGLE_CLIENT_ID || '',
            clientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
        }),
    ],
    pages: {
        signIn: '/login',
    },
    callbacks: {
        async jwt({ token, user }) {
            if (user) {
                token.role = user.role || 'agent'
            }
            return token
        },
        async session({ session, token }) {
            if (session.user) {
                session.user.id = token.sub
                session.user.role = token.role
            }
            return session
        },
        authorized({ auth, request: { nextUrl } }) {
            const isLoggedIn = !!auth?.user
            const isOnLogin = nextUrl.pathname.startsWith('/login')

            if (isOnLogin) {
                if (isLoggedIn) return Response.redirect(new URL('/manager/dashboard', nextUrl))
                return true
            }

            if (!isLoggedIn) {
                return false // Redirect to login
            }

            return true
        },
    },
})
