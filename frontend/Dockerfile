# ── Build stage ──────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build
RUN npm run build

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

# Install serve for static file serving
RUN npm install -g serve

# Copy built assets
COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
