# File: backend/app/api/routes/llm_trigger/__init__.py

## Purpose

Tiny package export file for the route module.

## What It Does

- Re-exports `router` from `router.py`.

## What You Can Modify

- Keep as-is unless you split llm trigger routes into multiple files.

## Risky Changes

- Import path changes without matching updates in `backend/app/api/main.py` will break route registration.
