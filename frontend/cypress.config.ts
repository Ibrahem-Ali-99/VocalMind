import coverageTask from '@cypress/code-coverage/task';
import { defineConfig } from 'cypress';

const coverageEnabled = process.env.CYPRESS_COVERAGE === 'true';

export default defineConfig({
  env: {
    coverage: coverageEnabled,
  },
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
      if (config.env.coverage) {
        coverageTask(on, config);
      }
      return config;
    },
  },
});
