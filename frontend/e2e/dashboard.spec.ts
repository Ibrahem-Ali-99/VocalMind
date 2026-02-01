import { test, expect } from '@playwright/test'

test.describe('Dashboard', () => {
    test.beforeEach(async ({ page }) => {
        // Login first
        await page.goto('/login')
        await page.fill('input[type="email"]', 'test@vocalmind.com')
        await page.fill('input[type="password"]', 'password')
        await page.click('button:has-text("Sign In")')
        await page.waitForURL(/dashboard/)
    })

    test('should display stats cards', async ({ page }) => {
        await expect(page.locator('text=Welcome back')).toBeVisible()
        await expect(page.locator('text=Total Calls Today')).toBeVisible()
        await expect(page.locator('text=Avg Sentiment Score')).toBeVisible()
    })

    test('should show flagged calls table', async ({ page }) => {
        await expect(page.locator('text=Flagged Calls')).toBeVisible()
        await expect(page.locator('text=PRIORITY')).toBeVisible()
    })

    test('should navigate to calls page', async ({ page }) => {
        await page.click('text=View All')
        await expect(page).toHaveURL(/calls/)
    })
})
