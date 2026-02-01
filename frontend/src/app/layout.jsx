import './globals.css'
import Shell from '@/components/Shell'
import { AuthProvider } from '@/components/providers/AuthProvider'

export const metadata = {
    title: 'VocalMind',
    description: 'AI-powered call center analytics',
    icons: {
        icon: '/favicon.png',
        apple: '/favicon.png',
    },
}

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body className="font-sans antialiased bg-gray-50 text-gray-900">
                <AuthProvider>
                    <Shell>{children}</Shell>
                </AuthProvider>
            </body>
        </html>
    )
}
