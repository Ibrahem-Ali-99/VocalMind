'use client'

import { memo } from 'react'

export const AISummaryCard = memo(({ summary, status }) => (
    <div className="bg-gradient-to-br from-cyan/10 to-purple/10 rounded-xl p-5 mb-4">
        <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-gray-900">AI Summary</h2>
            <span
                className={`px-2 py-0.5 text-xs font-medium rounded ${status === 'Resolved' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}
            >
                {status}
            </span>
        </div>
        <p className="text-sm text-gray-700 leading-relaxed">{summary}</p>
    </div>
))
AISummaryCard.displayName = 'AISummaryCard'

export const KeyTakeaways = memo(({ takeaways }) => (
    <div className="mb-4">
        <h2 className="font-semibold text-gray-900 mb-3">Key Takeaways</h2>
        <ul className="space-y-2">
            {takeaways.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                    <svg
                        className="w-4 h-4 text-cyan mt-0.5 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                    >
                        <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                        />
                    </svg>
                    {item}
                </li>
            ))}
        </ul>
    </div>
))
KeyTakeaways.displayName = 'KeyTakeaways'

export const SentimentDrivers = memo(({ drivers }) => (
    <div className="mb-4">
        <h2 className="font-semibold text-gray-900 mb-3">Sentiment Drivers</h2>
        <div className="space-y-2">
            {drivers.map((driver, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                    <span
                        className={`w-2 h-2 rounded-full ${driver.impact === 'positive' ? 'bg-emerald-500' : 'bg-red-500'}`}
                    />
                    <span className="text-gray-700 flex-1">{driver.trigger}</span>
                    <span className="text-xs text-gray-500">
                        {Math.floor(driver.time / 60)}:{(driver.time % 60).toString().padStart(2, '0')}
                    </span>
                </div>
            ))}
        </div>
    </div>
))
SentimentDrivers.displayName = 'SentimentDrivers'

export const ComplianceScore = memo(({ score, items }) => (
    <div className="mb-4 p-4 bg-gray-50 rounded-xl">
        <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-gray-900">Policy Compliance</h2>
            <span
                className={`text-lg font-bold ${score >= 90 ? 'text-emerald-600' : score >= 70 ? 'text-amber-600' : 'text-red-600'}`}
            >
                {score}%
            </span>
        </div>
        <div className="space-y-1.5">
            {items.map((item, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                    {item.passed ? (
                        <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                clipRule="evenodd"
                            />
                        </svg>
                    ) : (
                        <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                clipRule="evenodd"
                            />
                        </svg>
                    )}
                    <span className={item.passed ? 'text-gray-700' : 'text-gray-500'}>{item.item}</span>
                </div>
            ))}
        </div>
    </div>
))
ComplianceScore.displayName = 'ComplianceScore'

export const CoachingTips = memo(({ tips }) => (
    <div className="mb-4">
        <h2 className="font-semibold text-gray-900 mb-3">Suggested Coaching</h2>
        <div className="space-y-2">
            {tips.map((tip, i) => (
                <div
                    key={i}
                    className={`p-3 rounded-lg text-sm ${tip.category === 'Strength' ? 'bg-emerald-50 border border-emerald-100' : 'bg-amber-50 border border-amber-100'}`}
                >
                    <span
                        className={`text-xs font-medium ${tip.category === 'Strength' ? 'text-emerald-700' : 'text-amber-700'}`}
                    >
                        {tip.category}
                    </span>
                    <p className="text-gray-700 mt-1">{tip.tip}</p>
                </div>
            ))}
        </div>
    </div>
))
CoachingTips.displayName = 'CoachingTips'
