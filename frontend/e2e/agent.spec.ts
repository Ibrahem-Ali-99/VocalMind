import { test, expect } from '@playwright/test'

test.describe('Agent Dashboard', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/login')
        await page.fill('input[type="email"]', 'agent@vocalmind.com')
        await page.fill('input[type="password"]', 'password')
        await page.click('button:has-text("Sign In")')
        await page.waitForURL(/dashboard/)
    })

    test('should show agent dashboard', async ({ page }) => {
        await page.goto('/agent/dashboard')
        await expect(page.locator('text=Welcome back')).toBeVisible()
        await expect(page.locator('text=Calls Handled')).toBeVisible()
    })

    test('should show AI coaching tips', async ({ page }) => {
        await page.goto('/agent/dashboard')
        await expect(page.locator('text=AI Coaching Tips')).toBeVisible()
    })

    test('should navigate to performance page', async ({ page }) => {
        await page.goto('/agent/performance')
        await expect(page.locator('text=My Performance')).toBeVisible()
        await expect(page.locator('text=Skills Breakdown')).toBeVisible()
    })

    test('should navigate to training page', async ({ page }) => {
        await page.goto('/agent/training')
        await expect(page.locator('text=Training Resources')).toBeVisible()
    })
})
