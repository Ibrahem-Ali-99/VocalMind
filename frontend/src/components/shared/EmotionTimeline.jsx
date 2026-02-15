'use client'

import { useRef, useEffect, useState, useMemo, useCallback, memo } from 'react'
import { emotionColors, emotionTextColors, emotionEmojis } from '@/utils/emotionColors'

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** Format seconds → "m:ss" */
const fmtTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
}

/** hex "#aabbcc" → { r, g, b } */
const hexToRgb = (hex) => {
    const c = (hex || '#94a3b8').replace('#', '')
    return {
        r: parseInt(c.slice(0, 2), 16),
        g: parseInt(c.slice(2, 4), 16),
        b: parseInt(c.slice(4, 6), 16),
    }
}

/** Seeded pseudo-random for consistent waveform shapes */
const seededRandom = (seed) => {
    const x = Math.sin(seed * 9301 + 4927) * 49297
    return x - Math.floor(x)
}

// ─────────────────────────────────────────────
// Canvas Waveform — Single Track Design
// ─────────────────────────────────────────────

const PAD = { top: 20, bottom: 30, left: 0, right: 0 }
const AGENT_COLOR = '#6366f1' // Indigo
const CLIENT_COLOR = '#2dd4bf' // Teal

function drawTimeline(canvas, segments, duration, hoverTime) {
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)

    const plotW = w - PAD.left - PAD.right
    const plotH = h - PAD.top - PAD.bottom
    const cx = PAD.left
    const midY = PAD.top + plotH / 2

    if (!segments.length || duration <= 0) return

    // 1. Background Emotion Zones
    segments.forEach((seg) => {
        const x1 = cx + (seg.start / duration) * plotW
        const x2 = cx + (seg.end / duration) * plotW
        const width = Math.max(x2 - x1, 1)
        
        const rgb = hexToRgb(emotionColors[seg.emotion])
        // Opacity varying by confidence: 0.05 (low) -> 0.15 (high)
        const alpha = 0.05 + (seg.confidence * 0.15)
        
        ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`
        ctx.fillRect(x1, 0, width, h) // Full height background
    })

    // 2. Central Axis
    ctx.strokeStyle = '#e2e8f0'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(cx, midY)
    ctx.lineTo(cx + plotW, midY)
    ctx.stroke()

    // 3. Waveform Bars
    const totalBars = Math.min(Math.floor(plotW / 3), 300) // Slightly wider bars
    const gap = 1.5
    const barW = Math.max((plotW - (totalBars - 1) * gap) / totalBars, 2)
    const maxBarH = (plotH / 2) - 4

    for (let i = 0; i < totalBars; i++) {
        const t = (i / totalBars) * duration
        const seg = segments.find((s) => t >= s.start && t < s.end)
        
        if (!seg) continue

        const x = cx + i * (barW + gap)
        const isAgent = seg.speaker === 'agent'
        
        // Amplitude calculation
        const segP = (t - seg.start) / (seg.end - seg.start || 1)
        const envelope = Math.sin(segP * Math.PI)
        const micro = seededRandom(i * 13 + seg.start * 7)
        // Base amplitude modulated by confidence
        const amp = (0.3 + envelope * 0.5 + micro * 0.2) * (0.6 + seg.confidence * 0.4)
        
        const barH = Math.max(amp * maxBarH, 2)

        if (isAgent) {
            // Agent: Upward, Indigo, Rounded Top
            ctx.fillStyle = AGENT_COLOR
            ctx.beginPath()
            // roundRect(x, y, w, h, [tl, tr, br, bl])
            ctx.roundRect(x, midY - barH, barW, barH, [2, 2, 0, 0])
            ctx.fill()
        } else {
            // Customer: Downward, Teal, Sharp/Slightly Rounded Bottom
            ctx.fillStyle = CLIENT_COLOR
            ctx.beginPath()
            ctx.roundRect(x, midY, barW, barH, [0, 0, 2, 2])
            ctx.fill()
        }
    }

    // 4. Time Axis Labels
    ctx.font = '500 10px Inter, system-ui, sans-serif'
    ctx.fillStyle = '#94a3b8'
    ctx.textAlign = 'center'
    const tickCount = 8
    for (let i = 0; i <= tickCount; i++) {
        const t = (i / tickCount) * duration
        const x = cx + (i / tickCount) * plotW
        ctx.fillText(fmtTime(t), x, h - 6)
    }

    // 5. Hover Scrubber
    if (hoverTime !== null && hoverTime >= 0 && hoverTime <= duration) {
        const hx = cx + (hoverTime / duration) * plotW
        
        // Line
        ctx.strokeStyle = '#64748b'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(hx, 0)
        ctx.lineTo(hx, h - 20)
        ctx.stroke()

        // Time Label Tag
        const timeLabel = fmtTime(hoverTime)
        ctx.font = '600 10px Inter, system-ui, sans-serif'
        const tw = ctx.measureText(timeLabel).width + 12
        
        ctx.fillStyle = '#334155'
        ctx.beginPath()
        ctx.roundRect(hx - tw / 2, 0, tw, 20, 4)
        ctx.fill()
        
        ctx.fillStyle = '#fff'
        ctx.fillText(timeLabel, hx, 14)
    }
}

// ─────────────────────────────────────────────
// Hover Tooltip
// ─────────────────────────────────────────────

const Tooltip = memo(({ segment, x, y }) => {
    if (!segment) return null
    
    const isAgent = segment.speaker === 'agent'
    const color = isAgent ? AGENT_COLOR : CLIENT_COLOR
    const bgColor = isAgent ? 'bg-indigo-50' : 'bg-teal-50'
    const txtColor = isAgent ? 'text-indigo-700' : 'text-teal-700'
    
    const emotionColor = emotionColors[segment.emotion] || '#94a3b8'
    const emoji = emotionEmojis[segment.emotion] || '•'

    return (
        <div
            className="absolute z-50 pointer-events-none bg-white rounded-lg shadow-xl border border-gray-100 flex flex-col gap-2 p-3 min-w-[220px]"
            style={{ 
                left: x, 
                top: y, 
                transform: `translate(${x > 150 ? '-100%' : '0%'}, -100%)`, // Smart positioning 
                marginTop: -10,
                marginLeft: x > 150 ? -10 : 10
            }}
        >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-gray-50 pb-2">
                <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] uppercase tracking-wide font-bold ${bgColor} ${txtColor}`}>
                    {isAgent ? 'Agent' : 'Customer'}
                </div>
                <div className="flex items-center gap-1 text-xs font-medium text-gray-500">
                    <span style={{ color: emotionColor }}>{emoji}</span>
                    <span className="capitalize text-gray-700">{segment.emotion}</span>
                    <span className="text-gray-300">•</span>
                    <span>{Math.round(segment.confidence * 100)}%</span>
                </div>
            </div>

            {/* Text Preview */}
            <p className="text-xs text-gray-600 leading-relaxed line-clamp-3 italic">
                "{segment.text}"
            </p>
            
            {/* Time range */}
            <div className="text-[10px] text-gray-400 font-mono mt-0.5">
                {fmtTime(segment.start)} - {fmtTime(segment.end)}
            </div>
        </div>
    )
})
Tooltip.displayName = 'Tooltip'

// ─────────────────────────────────────────────
// Emotion Legend Chips
// ─────────────────────────────────────────────

const Legend = memo(({ segments }) => {
    const emotions = useMemo(() => {
        const seen = new Set()
        return segments
            .filter((s) => {
                if (seen.has(s.emotion)) return false
                seen.add(s.emotion)
                return true
            })
            .map((s) => s.emotion)
    }, [segments])

    return (
        <div className="flex items-center gap-2 flex-wrap">
            {emotions.map((e) => (
                <span
                    key={e}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium capitalize transition-colors duration-200"
                    style={{
                        backgroundColor: `${emotionColors[e] || '#94a3b8'}15`,
                        color: emotionTextColors[e] || '#374151',
                    }}
                >
                    <span>{emotionEmojis[e] || '•'}</span>
                    {e}
                </span>
            ))}
        </div>
    )
})
Legend.displayName = 'Legend'

// ─────────────────────────────────────────────
// Chat Bubble (Unchanged)
// ─────────────────────────────────────────────

const ChatBubble = memo(({ segment, isAgent, agentName }) => {
    const color = emotionColors[segment.emotion] || '#94a3b8'
    const textColor = emotionTextColors[segment.emotion] || '#374151'
    const emoji = emotionEmojis[segment.emotion] || '❓'

    const Meta = (
        <div className={`flex items-center gap-2 mb-1.5 text-xs ${isAgent ? '' : 'flex-row-reverse'}`}>
            <span className="font-bold text-gray-900">
                {isAgent ? agentName : 'Customer'}
            </span>
            <span className="text-gray-400 tabular-nums text-[11px]">
                {fmtTime(segment.start)}
            </span>
            <span
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold capitalize"
                style={{ backgroundColor: `${color}15`, color: textColor }}
            >
                <span>{emoji}</span>
                {segment.emotion}
            </span>
        </div>
    )

    return (
        <div className={`flex gap-3 ${isAgent ? '' : 'flex-row-reverse'} group`}>
            <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white shadow-sm flex-shrink-0 mt-1 ${
                    isAgent ? 'bg-indigo-600' : 'bg-teal-500' // Matches waveform colors
                }`}
            >
                {isAgent ? (agentName?.[0] || 'A') : 'C'}
            </div>

            <div
                className="w-1 self-stretch rounded-full opacity-60"
                style={{ backgroundColor: color }}
            />

            <div className={`flex-1 max-w-[85%] flex flex-col ${isAgent ? 'items-start' : 'items-end'}`}>
                {Meta}
                <div
                    className={`rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm border transition-shadow hover:shadow-md ${
                        isAgent
                            ? 'bg-white border-gray-100 text-gray-700 rounded-tl-none'
                            : 'bg-gray-50 border-gray-100 text-gray-800 rounded-tr-none'
                    }`}
                >
                    {segment.text}
                </div>
            </div>
        </div>
    )
})
ChatBubble.displayName = 'ChatBubble'

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────

export default function EmotionTimeline({ data }) {
    const canvasRef = useRef(null)
    const [hoverTime, setHoverTime] = useState(null)
    const [tooltipSeg, setTooltipSeg] = useState(null)
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

    const { meta, segments } = data
    const duration = meta.duration
    const sorted = useMemo(() => [...segments].sort((a, b) => a.start - b.start), [segments])

    // Draw Loop
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        
        const draw = () => {
            const ctx = canvas.getContext('2d')
            const dpr = window.devicePixelRatio || 1
            const w = canvas.clientWidth
            const h = canvas.clientHeight
            canvas.width = w * dpr
            canvas.height = h * dpr
            ctx.scale(dpr, dpr)
            ctx.clearRect(0, 0, w, h)

            const plotW = w - PAD.left - PAD.right
            const plotH = h - PAD.top - PAD.bottom
            const cx = PAD.left
            const midY = PAD.top + plotH / 2

            if (!segments.length || duration <= 0) return

            // 1. Background Emotion Zones (Subtle)
            sorted.forEach((seg) => {
                const x1 = cx + (seg.start / duration) * plotW
                const x2 = cx + (seg.end / duration) * plotW
                const width = Math.max(x2 - x1, 1)
                
                const rgb = hexToRgb(emotionColors[seg.emotion])
                // Even more subtle background for "white div" look
                const alpha = 0.03 + (seg.confidence * 0.07)
                
                ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`
                ctx.fillRect(x1, 0, width, h)
            })

            // 2. Central Axis (Very Light)
            ctx.strokeStyle = '#f1f5f9'
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(cx, midY)
            ctx.lineTo(cx + plotW, midY)
            ctx.stroke()

            // 3. Waveform Bars (Thinner & Denser)
            const totalBars = Math.floor(plotW / 2.5) // More bars
            const gap = 1
            const barW = (plotW - (totalBars - 1) * gap) / totalBars
            const maxBarH = (plotH / 2) - 4

            for (let i = 0; i < totalBars; i++) {
                const t = (i / totalBars) * duration
                const seg = sorted.find((s) => t >= s.start && t < s.end)
                
                if (!seg) continue

                const x = cx + i * (barW + gap)
                const isAgent = seg.speaker === 'agent'
                
                const segP = (t - seg.start) / (seg.end - seg.start || 1)
                const envelope = Math.sin(segP * Math.PI)
                const micro = seededRandom(i * 13 + seg.start * 7)
                const amp = (0.2 + envelope * 0.6 + micro * 0.2) * (0.6 + seg.confidence * 0.4)
                
                const barH = Math.max(amp * maxBarH, 1.5)

                ctx.fillStyle = isAgent ? AGENT_COLOR : CLIENT_COLOR
                if (isAgent) {
                    ctx.beginPath()
                    ctx.roundRect(x, midY - barH, barW, barH, [1, 1, 0, 0])
                    ctx.fill()
                } else {
                    ctx.beginPath()
                    ctx.roundRect(x, midY, barW, barH, [0, 0, 1, 1])
                    ctx.fill()
                }
            }

            // 4. Time Labels
            ctx.font = '500 10px Inter, system-ui, sans-serif'
            ctx.fillStyle = '#cbd5e1'
            ctx.textAlign = 'center'
            const tickCount = Math.floor(plotW / 80)
            for (let i = 0; i <= tickCount; i++) {
                const t = (i / tickCount) * duration
                const x = cx + (i / tickCount) * plotW
                ctx.fillText(fmtTime(t), x, h - 6)
            }

            // 5. Scrubber
            if (hoverTime !== null) {
                const hx = cx + (hoverTime / duration) * plotW
                ctx.strokeStyle = '#94a3b8'
                ctx.lineWidth = 1
                ctx.beginPath()
                ctx.moveTo(hx, 0)
                ctx.lineTo(hx, h - 20)
                ctx.stroke()

                const timeLabel = fmtTime(hoverTime)
                ctx.font = '600 10px Inter, system-ui, sans-serif'
                const tw = ctx.measureText(timeLabel).width + 12
                ctx.fillStyle = '#1e293b'
                ctx.beginPath()
                ctx.roundRect(hx - tw / 2, 0, tw, 20, 4)
                ctx.fill()
                ctx.fillStyle = '#fff'
                ctx.fillText(timeLabel, hx, 14)
            }
        }

        draw()
        const ro = new ResizeObserver(draw)
        ro.observe(canvas.parentElement)
        return () => ro.disconnect()
    }, [sorted, duration, hoverTime])

    // Mouse Interaction
    const handleMove = useCallback((e) => {
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const mx = e.clientX - rect.left
        const t = (mx / rect.width) * duration
        
        if (t < 0 || t > duration) {
            setHoverTime(null)
            setTooltipSeg(null)
            return
        }
        setHoverTime(t)
        const seg = sorted.find((s) => t >= s.start && t <= s.end)
        if (seg) {
            setTooltipSeg(seg)
            setTooltipPos({ x: mx, y: e.clientY - rect.top })
        } else {
            setTooltipSeg(null)
        }
    }, [sorted, duration])

    const handleLeave = useCallback(() => {
        setHoverTime(null)
        setTooltipSeg(null)
    }, [])

    return (
        <div className="w-full">
            {/* Waveform Card */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                {/* Header */}
                <div className="px-5 py-4 border-b border-gray-50 flex items-center justify-between bg-white">
                    <div>
                        <h2 className="font-semibold text-gray-900 text-sm tracking-tight">Timeline Analysis</h2>
                        <div className="flex items-center gap-2 mt-1">
                            <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                            <span className="text-xs text-gray-500 font-medium">{meta.agent_name}</span>
                            <span className="text-gray-300">|</span>
                            <span className="w-2 h-2 rounded-full bg-teal-400"></span>
                            <span className="text-xs text-gray-500 font-medium">{meta.client_name}</span>
                        </div>
                    </div>
                    <Legend segments={sorted} />
                </div>

                {/* Canvas Area */}
                <div className="relative h-[240px] w-full bg-white">
                    <canvas
                        ref={canvasRef}
                        className="w-full h-full cursor-crosshair block"
                        onMouseMove={handleMove}
                        onMouseLeave={handleLeave}
                    />
                    <Tooltip segment={tooltipSeg} x={tooltipPos.x} y={tooltipPos.y} />
                </div>
            </div>

            {/* Transcript */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 mt-6 overflow-hidden">
                <div className="px-5 py-4 border-b border-gray-50 bg-white">
                    <h2 className="font-semibold text-gray-900 text-sm">Transcript</h2>
                </div>
                <div className="px-5 py-6 space-y-6 max-h-[600px] overflow-y-auto custom-scrollbar bg-white">
                    {sorted.map((seg, i) => (
                        <ChatBubble
                            key={`${seg.start}-${i}`}
                            segment={seg}
                            isAgent={seg.speaker === 'agent'}
                            agentName={meta.agent_name.split(' ')[0]}
                        />
                    ))}
                </div>
            </div>
        </div>
    )
}

