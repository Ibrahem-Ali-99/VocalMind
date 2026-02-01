import { auth } from '@/auth'

export default auth((req) => {
    // This middleware runs for all protected routes
})

// Protect all routes except login and API
export const config = {
    matcher: ['/((?!api|_next/static|_next/image|favicon.ico|images|login).*)'],
}
