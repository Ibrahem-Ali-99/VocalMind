import React from 'react'

const StarRating = ({ score, maxStars = 5, size = 'md' }) => {
    const fullStars = Math.floor(score)
    const hasHalfStar = score % 1 >= 0.5
    const emptyStars = maxStars - fullStars - (hasHalfStar ? 1 : 0)

    const sizeClasses = {
        sm: 'w-3 h-3',
        md: 'w-4 h-4',
        lg: 'w-5 h-5',
    }
    const iconSize = sizeClasses[size] || sizeClasses.md

    const getScoreColor = (score) => {
        if (score >= 4.0) return 'text-emerald-500'
        if (score >= 3.0) return 'text-amber-500'
        return 'text-red-500'
    }

    const colorClass = getScoreColor(score)

    return (
        <div className="flex items-center gap-2">
            <div className="flex items-center">
                {[...Array(fullStars)].map((_, i) => (
                    <svg
                        key={`full-${i}`}
                        className={`${iconSize} ${colorClass}`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                    >
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                ))}
                {hasHalfStar && (
                    <svg className={`${iconSize} ${colorClass}`} fill="currentColor" viewBox="0 0 20 20">
                        <defs>
                            <linearGradient id={`halfGradient-${score}`}>
                                <stop offset="50%" stopColor="currentColor" />
                                <stop offset="50%" stopColor="#E5E7EB" />
                            </linearGradient>
                        </defs>
                        <path
                            fill={`url(#halfGradient-${score})`}
                            d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
                        />
                    </svg>
                )}
                {[...Array(emptyStars)].map((_, i) => (
                    <svg
                        key={`empty-${i}`}
                        className={`${iconSize} text-gray-200`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                    >
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                ))}
            </div>
            <span className={`text-sm font-medium ${colorClass}`}>{score.toFixed(1)}</span>
        </div>
    )
}

export default StarRating
