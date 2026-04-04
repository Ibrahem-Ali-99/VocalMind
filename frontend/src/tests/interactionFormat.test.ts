import { describe, expect, it } from 'vitest'

import { formatResponseTime, parseInteractionDateTime } from '../app/utils/interactionFormat'

describe('interactionFormat', () => {
    describe('formatResponseTime', () => {
        it.each([
            [undefined, 'N/A'],
            [null, 'N/A'],
            ['', 'N/A'],
            ['   ', 'N/A'],
            ['N/A', 'N/A'],
            ['  n/a  ', 'N/A'],
            ['1.3s', '1.3s'],
            [' 1.3 ', '1.3s'],
        ])('formats %p as %p', (input, expected) => {
            expect(formatResponseTime(input as string | null | undefined)).toBe(expected)
        })
    })

    describe('parseInteractionDateTime', () => {
        it.each([
            ['2026-03-21', '10:15 AM', new Date(2026, 2, 21, 10, 15).getTime()],
            ['2026-03-21', '10:15 PM', new Date(2026, 2, 21, 22, 15).getTime()],
            ['2026-03-21', '12:00 AM', new Date(2026, 2, 21, 0, 0).getTime()],
            ['2026-03-21', '12:00 PM', new Date(2026, 2, 21, 12, 0).getTime()],
            ['2026-03-21', '17:45', new Date(2026, 2, 21, 17, 45).getTime()],
            ['2026-03-21', 'bad time', new Date(2026, 2, 21).getTime()],
        ])('parses %s %s into the expected timestamp', (date, time, expected) => {
            expect(parseInteractionDateTime(date, time)).toBe(expected)
        })

        it('returns zero when the date is invalid', () => {
            expect(parseInteractionDateTime('bad-date', '10:15 AM')).toBe(0)
        })
    })
})
