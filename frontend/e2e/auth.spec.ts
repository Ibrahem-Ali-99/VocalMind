import { test, expect } from '@playwright/test'

test.describe('Authentication', () => {
    test('should show login page', async ({ page }) => {
        await page.goto('/login')

        // Use more specific locator - target the main login heading
        await expect(page.getByRole('heading', { name: 'VocalMind' })).toBeVisible()
        await expect(page.locator('input[type="email"]')).toBeVisible()
        await expect(page.locator('input[type="password"]')).toBeVisible()
        await expect(page.getByRole('button', { name: 'Sign In' })).toBeVisible()
    })

    test('should login with credentials', async ({ page }) => {
        await page.goto('/login')

        await page.fill('input[type="email"]', 'test@vocalmind.com')
        await page.fill('input[type="password"]', 'password123')
        await page.click('button:has-text("Sign In")')

        // Should redirect to dashboard
        await expect(page).toHaveURL(/\/manager\/dashboard/)
    })

    test('should show SSO options', async ({ page }) => {
        await page.goto('/login')

        await expect(page.getByRole('button', { name: 'Google' })).toBeVisible()
        await expect(page.getByRole('button', { name: 'Microsoft' })).toBeVisible()
    })
})
