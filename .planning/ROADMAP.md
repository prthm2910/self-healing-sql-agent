# Roadmap: Session-Based & Persistent Chatbot

## Overview
A conversational AI application using Streamlit for the frontend, LangGraph for orchestration, and Gemini 1.5 models. Implements a two-layer memory system: short-term thread persistence and long-term semantic user memory via Neon DB (PostgreSQL).

## Phases

- [x] **Phase 1: Project Foundation** - Environment setup and secret management.
- [x] **Phase 2: Core Chat Interface** - Functional chat UI connected to Gemini.
- [x] **Phase 3: Persistent Memory Upgrade** - Implementation of Postgres-backed thread memory and semantic long-term storage with background reflection.
- [ ] **Phase 4: Memory Refinement** - Refine the memory pipeline to ensure 'Atomic Fact Retention' (deterministic storage and explicit extraction).
- [ ] **Phase 5: Unified Logging Standard** - Establish a unified, contextual logging standard across all layers of the application.
- [ ] **Phase 6: Divide & Conquer Decomposition** - Implement a decomposition workflow to handle complex queries by breaking them into sub-tasks.

## Progress

| Phase | Status | Completed |
|-------|--------|-----------|
| 1. Project Foundation | Completed | 2026-04-27 |
| 2. Core Chat Interface | Completed | 2026-04-27 |
| 3. Persistent Memory Upgrade | Completed | 2026-04-27 |
| 4. Memory Refinement | In Progress | - |
| 5. Unified Logging Standard | Planned | - |
| 6. Divide & Conquer | Planned | - |

## Future Enhancements
- [ ] User authentication for multi-user support.
- [ ] Support for multimodal inputs (images/PDFs) in the memory reflector.
- [ ] Automated memory pruning or consolidation.

## Phase 4 Plans
- [ ] 04-01-PLAN.md - Atomic Fact Retention implementation

## Phase 5 Plans
**Plans:** 2 plans
- [ ] 05-01-PLAN.md - Logging Infrastructure & Service Core
- [ ] 05-02-PLAN.md - Service Expansion & Log Audit

## Phase 6 Plans
**Plans:** 3 plans
- [ ] 06-01-PLAN.md - Foundation & SQLTranspiler
- [ ] 06-02-PLAN.md - Manager & Worker Nodes
- [ ] 06-03-PLAN.md - Assembler & Graph Orchestration
