// VocalMind emotion palette — aligned to the 7 model classes
// Model: j-hartmann/emotion-english-distilroberta-base
// Labels: anger, disgust, fear, joy, neutral, sadness, surprise

// Vibrant background / fill colors (modern, aesthetic)
export const emotionColors = {
    anger:    '#ef4444',   // warm red
    disgust:  '#a3e635',   // lime green
    fear:     '#facc15',   // golden amber
    joy:      '#34d399',   // emerald mint
    neutral:  '#94a3b8',   // cool slate
    sadness:  '#60a5fa',   // soft sky blue
    surprise: '#c084fc',   // lavender purple

    // Agent-specific / extended labels (fallback to closest model class)
    frustrated: '#f97316', // orange → close to anger
    anxious:    '#fbbf24', // amber → close to fear
    hopeful:    '#34d399', // emerald → close to joy
    happy:      '#10b981', // green → synonym for joy
    grateful:   '#2dd4bf', // teal → positive
    empathetic: '#818cf8', // indigo → calm supportive
    confident:  '#6366f1', // deep indigo
    helpful:    '#a78bfa', // soft violet
    satisfied:  '#22c55e', // green → positive
}

// Darker text-safe variants (meets WCAG AA contrast)
export const emotionTextColors = {
    anger:      '#b91c1c',
    disgust:    '#4d7c0f',
    fear:       '#a16207',
    joy:        '#047857',
    neutral:    '#334155',
    sadness:    '#1d4ed8',
    surprise:   '#7c3aed',

    frustrated: '#c2410c',
    anxious:    '#92400e',
    hopeful:    '#047857',
    happy:      '#065f46',
    grateful:   '#0f766e',
    empathetic: '#4338ca',
    confident:  '#4338ca',
    helpful:    '#6d28d9',
    satisfied:  '#15803d',
}

// Emoji map for graph labels
export const emotionEmojis = {
    anger:      '😡',
    disgust:    '🤢',
    fear:       '😨',
    joy:        '😊',
    neutral:    '😐',
    sadness:    '😢',
    surprise:   '😲',

    frustrated: '😤',
    anxious:    '😰',
    hopeful:    '🤞',
    happy:      '😄',
    grateful:   '🙏',
    empathetic: '🤝',
    confident:  '💪',
    helpful:    '🤗',
    satisfied:  '😌',
}

// Sentiment valence: 0 = most negative, 1 = most positive
// Used to position dots on the sentiment curve
export const sentimentValue = {
    anger:      0.05,
    disgust:    0.1,
    fear:       0.15,
    sadness:    0.2,
    frustrated: 0.2,
    anxious:    0.3,
    neutral:    0.5,
    hopeful:    0.65,
    surprise:   0.7,
    helpful:    0.75,
    empathetic: 0.75,
    confident:  0.8,
    grateful:   0.85,
    satisfied:  0.85,
    joy:        0.9,
    happy:      0.95,
}
