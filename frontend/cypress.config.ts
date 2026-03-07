import coverageTask from '@cypress/code-coverage/task';
import { defineConfig } from 'cypress';

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    specPattern: 'cypress/e2e/**/*.cy.ts',
    supportFile: 'cypress/support/e2e.ts',
    
    // Default viewport optimized for desktop layouts
    viewportWidth: 1280,
    viewportHeight: 720,
    
    // Disable video to improve test suite performance
    video: false,
    
    setupNodeEvents(on, config) {
      // Register Cypress code coverage task
      coverageTask(on, config);
      return config;
    },
  },
});
