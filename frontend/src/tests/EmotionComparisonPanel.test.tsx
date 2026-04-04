import React from 'react'
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const recordedChartData: Array<Array<{ name: string; acoustic: number; text: number; fused: number }>> = []

vi.mock('recharts', () => ({
    ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
    CartesianGrid: () => null,
    Tooltip: () => null,
    Legend: () => null,
    XAxis: () => null,
    YAxis: () => null,
    Bar: ({ name }: any) => <div>{name}</div>,
    BarChart: ({ data, children }: any) => {
        recordedChartData.push(data)
        return (
            <div data-testid="emotion-bar-chart">
                <div data-testid="chart-labels">{data?.map((item: any) => item.name).join('|')}</div>
                {children}
            </div>
        )
    },
}))

import { EmotionComparisonPanel, type EmotionComparisonData } from '../app/components/manager/EmotionComparisonPanel'

function buildData(overrides: Partial<EmotionComparisonData> = {}): EmotionComparisonData {
    return {
        totalUtterances: overrides.totalUtterances ?? 12,
        distributions: overrides.distributions ?? {
            acoustic: [
                { emotion: 'happy', count: 4, pct: 33.3 },
                { emotion: 'neutral', count: 3, pct: 25 },
            ],
            text: [
                { emotion: 'happy', count: 3, pct: 25 },
                { emotion: 'frustrated', count: 2, pct: 16.7 },
            ],
            fused: [
                { emotion: 'happy', count: 5, pct: 41.7 },
                { emotion: 'empathetic', count: 2, pct: 16.7 },
            ],
        },
        quality: overrides.quality ?? {
            acousticTextAgreementRate: 88,
            fusedMatchesAcousticRate: 71,
            fusedMatchesTextRate: 52,
            disagreementCount: 0,
        },
    }
}

describe('EmotionComparisonPanel', () => {
    beforeEach(() => {
        recordedChartData.length = 0
    })

    it.each([
        [88, 'Excellent agreement'],
        [65, 'Good agreement'],
        [45, 'Fair agreement'],
        [15, 'Poor agreement'],
    ])('maps %s%% acoustic/text agreement to the "%s" quality copy', (rate, expectedLabel) => {
        render(
            <EmotionComparisonPanel
                data={buildData({
                    quality: {
                        acousticTextAgreementRate: rate,
                        fusedMatchesAcousticRate: 71,
                        fusedMatchesTextRate: 52,
                        disagreementCount: 0,
                    },
                })}
            />
        )

        expect(screen.getByText(`${rate.toFixed(1)}%`)).toBeInTheDocument()
        expect(screen.getByText(expectedLabel)).toBeInTheDocument()
    })

    it.each([
        [1, '1 utterance show mismatch between acoustic and text emotions.'],
        [3, '3 utterances show mismatch between acoustic and text emotions.'],
    ])('renders disagreement messaging for %s mismatched utterance(s)', (count, expectedCopy) => {
        render(
            <EmotionComparisonPanel
                data={buildData({
                    quality: {
                        acousticTextAgreementRate: 88,
                        fusedMatchesAcousticRate: 71,
                        fusedMatchesTextRate: 52,
                        disagreementCount: count,
                    },
                })}
            />
        )

        expect(screen.getByText('Cross-Modal Disagreement')).toBeInTheDocument()
        expect(screen.getByText(expectedCopy)).toBeInTheDocument()
    })

    it('aggregates and limits chart data to the six most frequent emotions across modalities', () => {
        render(
            <EmotionComparisonPanel
                data={buildData({
                    distributions: {
                        acoustic: [
                            { emotion: 'neutral', count: 9, pct: 18 },
                            { emotion: 'happy', count: 8, pct: 16 },
                            { emotion: 'frustrated', count: 7, pct: 14 },
                            { emotion: 'calm', count: 6, pct: 12 },
                        ],
                        text: [
                            { emotion: 'sad', count: 7, pct: 14 },
                            { emotion: 'angry', count: 6, pct: 12 },
                            { emotion: 'surprise', count: 5, pct: 10 },
                            { emotion: 'fear', count: 1, pct: 2 },
                        ],
                        fused: [
                            { emotion: 'happy', count: 4, pct: 8 },
                            { emotion: 'empathetic', count: 5, pct: 10 },
                            { emotion: 'disgust', count: 1, pct: 2 },
                        ],
                    },
                })}
            />
        )

        const latestChartData = recordedChartData[recordedChartData.length - 1] ?? []
        expect(latestChartData.map((item) => item.name)).toEqual([
            'Happy',
            'Neutral',
            'Frustrated',
            'Sad',
            'Calm',
            'Angry',
        ])
        expect(screen.getByTestId('chart-labels')).toHaveTextContent('Happy|Neutral|Frustrated|Sad|Calm|Angry')
        expect(screen.getByTestId('chart-labels')).not.toHaveTextContent('Fear')
        expect(screen.getByTestId('chart-labels')).not.toHaveTextContent('Disgust')
    })

    it('falls back to the neutral swatch color for emotions outside the known palette', () => {
        render(
            <EmotionComparisonPanel
                data={buildData({
                    distributions: {
                        acoustic: [{ emotion: 'curious', count: 2, pct: 20 }],
                        text: [{ emotion: 'curious', count: 1, pct: 10 }],
                        fused: [{ emotion: 'curious', count: 3, pct: 30 }],
                    },
                })}
            />
        )

        const acousticLabel = screen.getAllByText('curious')[0]
        const swatch = acousticLabel.previousElementSibling as HTMLElement

        expect(screen.getByText('2 (20.0%)')).toBeInTheDocument()
        expect(swatch).toHaveStyle({ backgroundColor: '#9CA3AF' })
    })
})
