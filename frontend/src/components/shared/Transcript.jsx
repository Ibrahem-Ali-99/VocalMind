'use client'

import { emotionColors, emotionTextColors } from '@/utils/emotionColors'

export default function Transcript({ messages, currentTime, onMessageClick }) {
    return (
        <div className="flex flex-col h-full bg-white rounded-xl shadow-card overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100 bg-gray-50">
                <h2 className="font-semibold text-gray-900">Full Transcript</h2>
                <p className="text-sm text-gray-600 mt-0.5">{messages.length} messages</p>
            </div>
            <div
                className="flex-1 overflow-y-auto p-4 space-y-4"
                tabIndex={0}
                role="region"
                aria-label="Transcript List"
            >
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        className={`cursor-pointer transition-all ${currentTime >= msg.time ? 'opacity-100 bg-cyan/5' : 'opacity-100 hover:bg-gray-50'}`}
                        onClick={() => onMessageClick(msg.time)}
                    >
                        <div className="flex items-center gap-2 mb-1">
                            <span
                                className={`text-xs font-medium capitalize ${msg.speaker === 'agent' ? 'text-purple' : 'text-cyan-800'}`}
                            >
                                {msg.speaker}
                            </span>
                            <span className="text-xs text-gray-500">
                                {Math.floor(msg.time / 60)}:{(msg.time % 60).toString().padStart(2, '0')}
                            </span>
                            <span
                                className="text-xs px-1.5 py-0.5 rounded capitalize"
                                style={{
                                    backgroundColor: `${emotionColors[msg.emotion]}20`,
                                    color: emotionTextColors[msg.emotion] || '#374151',
                                }}
                            >
                                {msg.emotion}
                            </span>
                        </div>
                        <p className="text-sm text-gray-700 leading-relaxed">{msg.text}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
