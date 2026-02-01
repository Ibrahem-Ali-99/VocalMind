'use client'

import React, { useMemo, memo } from 'react'
import { emotionColors } from '@/utils/emotionColors'

const WaveformLayer = memo(({ bars, active }) => (
    <div className="flex items-center gap-0.5 w-full h-full">
        {bars.map((bar, i) => (
            <div
                key={i}
                className="flex-1 rounded-full transition-all"
                style={{
                    height: `${bar.height}%`,
                    backgroundColor: active ? bar.color : `${bar.color}40`,
                    opacity: active ? 1 : 0.5,
                }}
            />
        ))}
    </div>
))
WaveformLayer.displayName = 'WaveformLayer'

export default function WaveformPlayer({
    duration,
    currentTime,
    emotionTimeline,
    onSeek,
    isPlaying,
    onPlayPause,
    playbackSpeed,
    onSpeedChange,
}) {
    const barsCount = 80

    const barsData = useMemo(() => {
        return Array.from({ length: barsCount }).map((_, i) => {
            const barTime = (i / barsCount) * duration
            const emotion = emotionTimeline.find((e, idx) => {
                const next = emotionTimeline[idx + 1]
                return barTime >= e.time && (!next || barTime < next.time)
            })
            const color = emotion ? emotionColors[emotion.customerEmotion] || '#9CA3AF' : '#9CA3AF'
            const noise = Math.sin(i * 132.1) * 30
            const height = Math.max(20, Math.min(100, 30 + Math.sin(i * 0.5) * 20 + Math.abs(noise)))

            return { height, color }
        })
    }, [duration, emotionTimeline])

    const progressPercent = Math.min(100, Math.max(0, (currentTime / duration) * 100))

    return (
        <div className="bg-white rounded-xl shadow-card p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="font-semibold text-gray-900">Audio Waveform</h2>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-600">Speed:</span>
                    {[0.5, 1, 1.5, 2].map((speed) => (
                        <button
                            key={speed}
                            onClick={() => onSpeedChange(speed)}
                            className={`px-2 py-1 text-xs rounded ${playbackSpeed === speed ? 'bg-cyan text-navy' : 'bg-gray-100 text-gray-600'}`}
                        >
                            {speed}x
                        </button>
                    ))}
                </div>
            </div>

            <div
                className="relative h-24 cursor-pointer group"
                onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect()
                    const x = e.clientX - rect.left
                    const percent = x / rect.width
                    onSeek(percent * duration)
                }}
            >
                <div className="absolute inset-0">
                    <WaveformLayer bars={barsData} active={false} />
                </div>

                <div
                    className="absolute inset-0 transition-[clip-path] duration-75 ease-linear will-change-[clip-path]"
                    style={{
                        clipPath: `inset(0 ${100 - progressPercent}% 0 0)`,
                        WebkitClipPath: `inset(0 ${100 - progressPercent}% 0 0)`,
                    }}
                >
                    <div className="w-full h-full">
                        <WaveformLayer bars={barsData} active={true} />
                    </div>
                </div>

                <div
                    className="absolute top-0 bottom-0 w-0.5 bg-navy transition-all duration-75 ease-linear will-change-[left]"
                    style={{ left: `${progressPercent}%` }}
                >
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-navy rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
            </div>

            <div className="relative h-6 mt-2">
                {emotionTimeline
                    .filter((e) => e.intensity >= 0.6)
                    .map((marker, i) => (
                        <div
                            key={i}
                            className="absolute transform -translate-x-1/2"
                            style={{ left: `${(marker.time / duration) * 100}%` }}
                        >
                            <div
                                className="w-2 h-2 rounded-full bg-orange-500"
                                title={`High Intensity: ${marker.customerEmotion}`}
                            />
                        </div>
                    ))}
            </div>

            <div className="flex items-center gap-4 mt-4">
                <button
                    onClick={onPlayPause}
                    className="w-12 h-12 rounded-full bg-cyan text-navy flex items-center justify-center hover:bg-cyan-light transition-all active:scale-95"
                    aria-label={isPlaying ? 'Pause' : 'Play'}
                >
                    {isPlaying ? (
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                        </svg>
                    ) : (
                        <svg className="w-6 h-6 ml-1" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M8 5v14l11-7z" />
                        </svg>
                    )}
                </button>
                <div className="flex-1">
                    <div
                        className="h-2 bg-gray-200 rounded-full overflow-hidden cursor-pointer"
                        onClick={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect()
                            const x = e.clientX - rect.left
                            onSeek((x / rect.width) * duration)
                        }}
                    >
                        <div
                            className="h-full bg-cyan rounded-full transition-all duration-75 ease-linear will-change-[width]"
                            style={{ width: `${progressPercent}%` }}
                        />
                    </div>
                </div>
                <span className="text-sm text-gray-600 font-mono tabular-nums">
                    {Math.floor(currentTime / 60)}:
                    {Math.floor(currentTime % 60)
                        .toString()
                        .padStart(2, '0')}{' '}
                    / {Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')}
                </span>
            </div>

            <div className="flex flex-wrap gap-3 mt-4 pt-4 border-t border-gray-100">
                {['angry', 'frustrated', 'anxious', 'neutral', 'hopeful', 'happy'].map((emotion) => (
                    <div key={emotion} className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded" style={{ backgroundColor: emotionColors[emotion] }} />
                        <span className="text-xs text-gray-600 capitalize">{emotion}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}
