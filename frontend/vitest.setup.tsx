import '@testing-library/jest-dom'
import * as matchers from '@testing-library/jest-dom/matchers'
import { vi, expect } from 'vitest'
import React from 'react'

expect.extend(matchers)

if (!HTMLElement.prototype.scrollIntoView) {
    HTMLElement.prototype.scrollIntoView = () => {}
}

// Mock Lucide Icons
vi.mock('lucide-react', () => {
    const makeIcon = (iconName: string) => () => <span data-testid={`icon-${iconName.toLowerCase()}`} />
    return new Proxy(
        {},
        {
            get(_target, prop) {
                if (prop === 'then') {
                    return undefined
                }
                if (prop === '__esModule') {
                    return true
                }
                if (typeof prop !== 'string') {
                    return undefined
                }
                return makeIcon(prop)
            },
            has(_target, prop) {
                return typeof prop === 'string' && prop !== 'then'
            },
            getOwnPropertyDescriptor(_target, prop) {
                if (typeof prop !== 'string' || prop === 'then') {
                    return undefined
                }
                return {
                    configurable: true,
                    enumerable: true,
                    value: makeIcon(prop),
                    writable: false,
                }
            },
        }
    )
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
