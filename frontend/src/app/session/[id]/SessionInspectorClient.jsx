'use client'

import { useState, useCallback, useRef, useEffect, lazy, Suspense } from 'react'
import { useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import Transcript from '@/components/shared/Transcript'
import LoadingSpinner from '@/components/shared/LoadingSpinner'
// Dynamic imports for heavy components
const WaveformPlayer = dynamic(() => import('@/components/shared/WaveformPlayer'), { ssr: false })

import {
    AISummaryCard,
    KeyTakeaways,
    SentimentDrivers,
    ComplianceScore,
    CoachingTips,
} from '@/components/shared/AnalysisComponents'

// Mock session data with comprehensive details
const mockSession = {
    id: 'CALL-2847',
    date: 'Jan 31, 2026',
    time: '2:34 PM',
    duration: '8:45',
    durationSeconds: 525,
    overallScore: 4.2,
    status: 'Resolved',
    riskLevel: 'Medium Risk',
    agent: {
        name: 'Ahmed Hassan',
        id: 'AGT-042',
        avatar: 'AH',
        performance: 4.5,
    },
    customer: {
        name: 'John Mitchell',
        id: 'CUST-8294',
        plan: 'Enterprise',
        tenure: '2 years',
    },
    transcript: [
        {
            speaker: 'agent',
            text: 'Thank you for calling VocalMind support. My name is Ahmed, how may I assist you today?',
            time: 0,
            emotion: 'neutral',
        },
        {
            speaker: 'customer',
            text: "Hi Ahmed. I've been having issues with my account for the past week. Every time I try to access my dashboard, I get an error.",
            time: 15,
            emotion: 'frustrated',
        },
        {
            speaker: 'agent',
            text: "I'm sorry to hear you've been experiencing difficulties. Let me look into this right away. Can you provide me with your account ID?",
            time: 35,
            emotion: 'empathetic',
        },
        {
            speaker: 'customer',
            text: "It's CUST-8294. I really need this resolved today because I have a presentation tomorrow.",
            time: 52,
            emotion: 'anxious',
        },
        {
            speaker: 'agent',
            text: 'I completely understand the urgency. I can see your account here, and I notice there was a configuration update last week that may have affected your access.',
            time: 70,
            emotion: 'confident',
        },
        {
            speaker: 'customer',
            text: 'That makes sense. The timing matches up. Can you fix it?',
            time: 95,
            emotion: 'hopeful',
        },
        {
            speaker: 'agent',
            text: "Absolutely. I'm resetting your access permissions now. This will take just a moment.",
            time: 108,
            emotion: 'helpful',
        },
        {
            speaker: 'customer',
            text: "Thank you so much! You've been really helpful.",
            time: 130,
            emotion: 'grateful',
        },
        {
            speaker: 'agent',
            text: "You're welcome! Your access should be restored now. Is there anything else I can help you with today?",
            time: 145,
            emotion: 'satisfied',
        },
        {
            speaker: 'customer',
            text: "No, that's all. Thank you again Ahmed!",
            time: 165,
            emotion: 'happy',
        },
    ],
    emotionTimeline: [
        { time: 0, customerEmotion: 'neutral', agentEmotion: 'neutral', intensity: 0.5 },
        { time: 15, customerEmotion: 'frustrated', agentEmotion: 'neutral', intensity: 0.7 },
        { time: 35, customerEmotion: 'frustrated', agentEmotion: 'empathetic', intensity: 0.6 },
        { time: 52, customerEmotion: 'anxious', agentEmotion: 'empathetic', intensity: 0.8 },
        { time: 70, customerEmotion: 'anxious', agentEmotion: 'confident', intensity: 0.6 },
        { time: 95, customerEmotion: 'hopeful', agentEmotion: 'confident', intensity: 0.4 },
        { time: 108, customerEmotion: 'hopeful', agentEmotion: 'helpful', intensity: 0.3 },
        { time: 130, customerEmotion: 'grateful', agentEmotion: 'satisfied', intensity: 0.2 },
        { time: 165, customerEmotion: 'happy', agentEmotion: 'satisfied', intensity: 0.1 },
    ],
    aiSummary:
        "Customer reported dashboard access issues following a recent system update. Agent identified the root cause as a permissions configuration change and successfully restored access. Customer's issue was fully resolved within the call duration.",
    keyTakeaways: [
        "Issue caused by last week's configuration update affecting user permissions",
        'Agent demonstrated excellent empathy during high-stress customer moment',
        'Quick resolution maintained customer trust and satisfaction',
        'Customer confirmed access was restored successfully',
        'Enterprise customer retained with positive experience',
    ],
    sentimentDrivers: [
        { trigger: 'Week-long access issues', impact: 'negative', time: 15 },
        { trigger: 'Agent acknowledgment of urgency', impact: 'positive', time: 35 },
        { trigger: 'Presentation deadline pressure', impact: 'negative', time: 52 },
        { trigger: 'Agent identified root cause quickly', impact: 'positive', time: 70 },
        { trigger: 'Immediate resolution action', impact: 'positive', time: 108 },
    ],
    complianceScore: 92,
    complianceItems: [
        { item: 'Proper greeting and name introduction', passed: true },
        { item: 'Account verification before changes', passed: true },
        { item: 'Explained issue cause to customer', passed: true },
        { item: 'Offered additional assistance', passed: true },
        { item: 'Case documentation completed', passed: false },
    ],
    coachingTips: [
        {
            category: 'Strength',
            tip: 'Excellent active listening and empathy shown throughout the call',
        },
        {
            category: 'Strength',
            tip: 'Quick problem identification demonstrated strong product knowledge',
        },
        {
            category: 'Improvement',
            tip: 'Consider documenting case notes during the call rather than after',
        },
        {
            category: 'Improvement',
            tip: 'Could offer proactive tips to prevent similar issues in future',
        },
    ],
}

// Score Badge Component
const ScoreBadge = ({ score }) => {
    const getColor = () => {
        if (score >= 4.5) return 'bg-emerald-100 text-emerald-800'
        if (score >= 3.5) return 'bg-cyan/20 text-cyan-900'
        if (score >= 2.5) return 'bg-amber-100 text-amber-800'
        return 'bg-red-100 text-red-800'
    }
    return (
        <span className={`px-3 py-1.5 rounded-full text-sm font-semibold ${getColor()}`}>
            {score.toFixed(1)} / 5.0
        </span>
    )
}

// Manager Feedback Form
const ManagerFeedback = () => {
    const [feedback, setFeedback] = useState('')
    const [rating, setRating] = useState(0)

    return (
        <div className="border-t border-gray-100 pt-4 mt-4">
            <h2 className="font-semibold text-gray-900 mb-3">Manager Feedback</h2>
            <div className="mb-3">
                <label className="text-sm text-gray-600 mb-2 block">Rating</label>
                <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                        <button
                            key={star}
                            onClick={() => setRating(star)}
                            className={`w-8 h-8 ${rating >= star ? 'text-amber-400' : 'text-gray-300'}`}
                            aria-label={`Rate ${star} out of 5 stars`}
                        >
                            <svg fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                        </button>
                    ))}
                </div>
            </div>
            <textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="Add feedback for the agent..."
                className="w-full p-3 border border-gray-200 rounded-lg text-sm resize-none focus:ring-2 focus:ring-cyan/50 focus:border-cyan outline-none"
                rows={3}
            />
            <button className="mt-3 w-full py-2 bg-cyan text-navy font-medium rounded-lg hover:bg-cyan-light btn-hover">
                Submit Feedback
            </button>
        </div>
    )
}

export default function SessionInspectorClient({ sessionId }) {
    const router = useRouter()
    const [currentTime, setCurrentTime] = useState(0)
    const [isPlaying, setIsPlaying] = useState(false)
    const [playbackSpeed, setPlaybackSpeed] = useState(1)

    const session = mockSession

    const handleSeek = useCallback((time) => {
        setCurrentTime(time)
    }, [])

    return (
        <div className="h-full flex flex-col -m-6">
            {/* Header */}
            <div className="px-6 py-4 bg-white border-b border-gray-200">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                        <button
                            onClick={() => router.push('/manager/calls')}
                            className="hover:text-cyan transition-colors"
                            aria-label="Back to Calls"
                        >
                            Calls
                        </button>
                        <span>›</span>
                        <span className="text-gray-900">{session.id}</span>
                    </div>
                    <div className="flex items-center gap-3">
                        <button className="px-4 py-2 text-gray-600 bg-white border border-gray-200 rounded-lg btn-hover flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
                                />
                            </svg>
                            Share
                        </button>
                        <button className="px-4 py-2 bg-cyan text-navy font-medium rounded-lg hover:bg-cyan-light btn-hover flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                                />
                            </svg>
                            Export
                        </button>
                    </div>
                </div>

                {/* Call Info Header */}
                <div className="mt-3 flex items-center gap-4">
                    <h1 className="text-xl font-semibold text-gray-900">Call #{session.id}</h1>
                    <ScoreBadge score={session.overallScore} />
                    <span className="px-2.5 py-1 bg-orange-100 text-orange-700 text-xs font-medium rounded-full">
                        {session.riskLevel}
                    </span>
                    <span className="px-2.5 py-1 bg-emerald-100 text-emerald-700 text-xs font-medium rounded-full">
                        {session.status}
                    </span>
                </div>
                <div className="mt-2 flex items-center gap-4 text-sm text-gray-600">
                    <span>
                        {session.date} • {session.time}
                    </span>
                    <span>•</span>
                    <span>Duration: {session.duration}</span>
                    <span>•</span>
                    <span>
                        Agent:{' '}
                        <span
                            className="text-cyan-800 hover:text-cyan-700 cursor-pointer font-medium"
                            onClick={() => router.push(`/manager/agent/${session.agent.id}`)}
                        >
                            {session.agent.name}
                        </span>
                    </span>
                    <span>•</span>
                    <span>
                        Customer: <span className="text-gray-700">{session.customer.id}</span>
                    </span>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col lg:flex-row lg:overflow-hidden overflow-y-auto lg:overflow-y-hidden">
                {/* Left - Transcript */}
                <div className="w-full lg:w-[360px] h-[500px] lg:h-auto border-b lg:border-b-0 lg:border-r border-gray-200 bg-white flex flex-col">
                    <Transcript
                        messages={session.transcript}
                        currentTime={currentTime}
                        onMessageClick={handleSeek}
                    />
                </div>

                {/* Center - Waveform & Metrics */}
                <div
                    className="flex-1 bg-gray-50 p-4 lg:p-6 overflow-y-auto"
                    tabIndex={0}
                    role="region"
                    aria-label="Audio Player and Metrics"
                >
                    <Suspense
                        fallback={
                            <div className="h-64 bg-white rounded-xl shadow-card flex items-center justify-center">
                                <LoadingSpinner size="lg" />
                            </div>
                        }
                    >
                        <WaveformPlayer
                            duration={session.durationSeconds}
                            currentTime={currentTime}
                            emotionTimeline={session.emotionTimeline}
                            onSeek={handleSeek}
                            isPlaying={isPlaying}
                            onPlayPause={() => setIsPlaying(!isPlaying)}
                            playbackSpeed={playbackSpeed}
                            onSpeedChange={setPlaybackSpeed}
                        />
                    </Suspense>

                    {/* Call Metrics Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
                        <div className="bg-white rounded-xl p-4 shadow-card">
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-8 h-8 rounded-lg bg-purple/10 flex items-center justify-center">
                                    <svg
                                        className="w-4 h-4 text-purple"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a2 2 0 01-2-2v-6a2 2 0 012-2h8z"
                                        />
                                    </svg>
                                </div>
                                <span className="text-xs text-gray-600">Talk Ratio</span>
                            </div>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-gray-900">62%</span>
                                <span className="text-xs text-gray-600 mb-1">Agent</span>
                            </div>
                            <div className="mt-2 h-2 bg-gray-100 rounded-full overflow-hidden flex">
                                <div className="h-full bg-purple" style={{ width: '62%' }} />
                                <div className="h-full bg-cyan" style={{ width: '38%' }} />
                            </div>
                            <div className="flex justify-between mt-1 text-xs text-gray-600">
                                <span>Agent 62%</span>
                                <span>Customer 38%</span>
                            </div>
                        </div>

                        <div className="bg-white rounded-xl p-4 shadow-card">
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center justify-center">
                                    <svg
                                        className="w-4 h-4 text-amber-600"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
                                        />
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2"
                                        />
                                    </svg>
                                </div>
                                <span className="text-xs text-gray-600">Silence</span>
                            </div>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-gray-900">8%</span>
                                <span className="text-xs text-gray-600 mb-1">of call</span>
                            </div>
                            <p className="text-xs text-gray-600 mt-2">~42 seconds total</p>
                        </div>

                        <div className="bg-white rounded-xl p-4 shadow-card">
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center">
                                    <svg
                                        className="w-4 h-4 text-red-500"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                        />
                                    </svg>
                                </div>
                                <span className="text-xs text-gray-600">Interruptions</span>
                            </div>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-gray-900">2</span>
                                <span className="text-xs text-gray-600 mb-1">times</span>
                            </div>
                            <p className="text-xs text-emerald-500 mt-2">✓ Below average</p>
                        </div>

                        <div className="bg-white rounded-xl p-4 shadow-card">
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-8 h-8 rounded-lg bg-cyan/10 flex items-center justify-center">
                                    <svg
                                        className="w-4 h-4 text-cyan-700"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                                        />
                                    </svg>
                                </div>
                                <span className="text-xs text-gray-600">Avg Response</span>
                            </div>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-gray-900">2.4s</span>
                            </div>
                            <p className="text-xs text-emerald-500 mt-2">✓ Fast response</p>
                        </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="flex flex-wrap items-center gap-3 mt-6 p-4 bg-white rounded-xl shadow-card">
                        <span className="text-sm font-medium text-gray-700 mr-2">Quick Actions:</span>
                        <button className="flex items-center gap-2 px-3 py-2 text-sm bg-amber-50 text-amber-700 rounded-lg hover:bg-amber-100 transition-all">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
                                />
                            </svg>
                            Flag for Review
                        </button>
                        <button className="flex items-center gap-2 px-3 py-2 text-sm bg-purple/10 text-purple rounded-lg hover:bg-purple/20 transition-all">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                                />
                            </svg>
                            Add to Training
                        </button>
                        <button className="flex items-center gap-2 px-3 py-2 text-sm bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-all">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                                />
                            </svg>
                            Escalate
                        </button>
                        <button className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all ml-auto">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                                />
                            </svg>
                            Download Audio
                        </button>
                    </div>
                </div>

                {/* Right - Analysis Panel */}
                <div
                    className="w-full lg:w-[380px] border-t lg:border-t-0 lg:border-l border-gray-200 bg-white overflow-y-auto p-5"
                    tabIndex={0}
                    role="region"
                    aria-label="Call Analysis"
                >
                    <AISummaryCard summary={session.aiSummary} status={session.status} />
                    <KeyTakeaways takeaways={session.keyTakeaways} />
                    <SentimentDrivers drivers={session.sentimentDrivers} />
                    <ComplianceScore score={session.complianceScore} items={session.complianceItems} />
                    <CoachingTips tips={session.coachingTips} />
                    <ManagerFeedback />
                </div>
            </div>
        </div>
    )
}
