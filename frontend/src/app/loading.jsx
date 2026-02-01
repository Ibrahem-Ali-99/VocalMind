import LoadingSpinner from '@/components/shared/LoadingSpinner'

export default function Loading() {
    return (
        <div className="flex h-screen items-center justify-center bg-gray-50">
            <LoadingSpinner size="lg" />
        </div>
    )
}
