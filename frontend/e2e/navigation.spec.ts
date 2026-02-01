import { test, expect } from '@playwright/test'

test.describe('Navigation', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/login')
        await page.fill('input[type="email"]', 'test@vocalmind.com')
        await page.fill('input[type="password"]', 'password')
        await page.click('button:has-text("Sign In")')
        await page.waitForURL(/dashboard/)
    })

    test('should navigate to calls page', async ({ page }) => {
        await page.click('text=Calls')
        await expect(page).toHaveURL(/calls/)
        await expect(page.locator('text=All Calls')).toBeVisible()
    })

    test('should navigate to team page', async ({ page }) => {
        await page.click('text=Team')
        await expect(page).toHaveURL(/team/)
    })

    test('should navigate to settings page', async ({ page }) => {
        await page.click('text=Settings')
        await expect(page).toHaveURL(/settings/)
        await expect(page.locator('text=Profile Settings')).toBeVisible()
    })

    test('should navigate to session inspector', async ({ page }) => {
        await page.goto('/session/CALL-2847')
        // Wait for loading to finish (SessionInspectorClient simulates 500ms load)
        await expect(page.locator('text=Call #CALL-2847')).toBeVisible({ timeout: 10000 })
    })
})
