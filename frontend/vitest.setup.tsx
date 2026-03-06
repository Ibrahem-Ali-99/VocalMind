import '@testing-library/jest-dom'
import * as matchers from '@testing-library/jest-dom/matchers'
import { vi, expect } from 'vitest'
import React from 'react'

expect.extend(matchers)

// Mock Lucide Icons
vi.mock('lucide-react', () => {
    const icons = [
        'BarChart2', 'Phone', 'CheckCircle', 'AlertTriangle', 'Star', 'TrendingUp', 'TrendingDown',
        'Target', 'Zap', 'Search', 'ArrowLeft', 'Play', 'ThumbsUp', 'ThumbsDown', 'XCircle', 'Flag',
        'BookOpen', 'HelpCircle', 'Info', 'ChevronDown'
    ]
    const mock: any = {}
    icons.forEach(icon => {
        mock[icon] = (props: any) => <div {...props} data-testid={`icon-${icon.toLowerCase()}`} />
    })
    return mock
})

// Mock UI components
vi.mock('./src/app/components/ui/switch', () => ({
    Switch: ({ checked, onCheckedChange }: any) => (
        <input type="checkbox" checked={checked} onChange={onCheckedChange} role="switch" />
    )
}))

// Mock ALL of recharts to prevent JSDOM layout issues
vi.mock('recharts', () => {
    const MockComponent = ({ children }: any) => <div>{children}</div>
    return {
        LineChart: MockComponent,
        Line: MockComponent,
        BarChart: MockComponent,
        Bar: MockComponent,
        XAxis: MockComponent,
        YAxis: MockComponent,
        CartesianGrid: MockComponent,
        Tooltip: MockComponent,
        Legend: MockComponent,
        PieChart: MockComponent,
        Pie: MockComponent,
        Cell: MockComponent,
        ResponsiveContainer: MockComponent,
    }
})
