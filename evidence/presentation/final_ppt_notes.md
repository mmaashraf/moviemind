# MovieMind Final Presentation Notes

This file is the speaking script and defense notes for the 7-slide template.
It is designed for:
- main talk flow (about 8-12 mins),
- a short Q&A defense (about 2 mins),
- implementation-focused follow-up.

---

## Slide 1 - Project Title

### On-slide content
- **MovieMind: End-to-End Movie Recommendation System**
- IIT Delhi CEP Final Project
- Scope: data to deployable AI system

### Speaker notes (what to say)
- "This project is not just a model demo; it is a full AI lifecycle implementation."
- "We built a recommendation pipeline from data wrangling, to ML/DL models, to API, web app, explainability, and evidence tracking."
- "A core goal was reproducibility and engineering hygiene, not just offline metric improvement."

### Why this matters
- Sets evaluator expectations: systems thinking, not notebook-only work.

---

## Slide 2 - Background

### On-slide content
- Recommenders are essential in OTT/e-commerce.
- Challenges: sparsity, long tail, cold start.
- Need: accurate + explainable + production-ready.

### Speaker notes
- "MovieLens-style rating data is sparse, which makes naive similarity methods unstable."
- "If we only optimize a metric without explainability and deployment, it is not industry-ready."
- "So we treated this as an end-to-end product engineering problem."

### Decision and rationale
- **Decision:** use a lifecycle framing instead of a single-model approach.
- **Based on:** project rubric and real-world requirements for traceability.
- **Outcome:** coherent deliverables across phases with audit-friendly evidence.

---

## Slide 3 - Problem Statement

### On-slide content
- Predict user-movie ratings and generate Top-N recommendations.
- Compare Baseline, ML, DL, and tuned DL.
- Provide transparent explanations and model inspection.
- Deliver via API + UI + NLP interface.

### Speaker notes
- "The business question is: what should a given user watch next?"
- "The engineering question is: can we serve this with multiple model families in one unified interface?"
- "The trust question is: can we explain why a recommendation appears?"

### Decision and rationale
- **Decision:** keep model registry architecture with family-agnostic endpoints.
- **Based on:** requirement to expose all model types in final app.
- **Outcome:** one API/UI flow can switch Baseline/ML/DL/Tuned-DL quickly.

---

## Slide 4 - Methodology, Datasets, and Tools

### On-slide content
- Dataset: MovieLens 1M (+ users/movies/ratings metadata)
- Split: strict time-based split
- Models: Baseline, Linear/RF/GB, NCF, Tuned NCF (Optuna)
- Stack: pandas, scikit-learn, PyTorch, FastAPI, Streamlit
- Explainability: feature importance, embedding maps, inspector

### Speaker notes
- "The most critical modeling decision was chronological split to avoid leakage."
- "If we train on future ratings and evaluate on past, accuracy looks better but is fake."
- "DL tuning was done with Optuna, while API/UI were built for runtime model switching."

### Decision and rationale
- **Decision:** time-based split over random split.
- **Based on:** recommendation systems are temporal by nature.
- **Outcome:** evaluation reflects real deployment scenario.

---

## Slide 5 - Results 1 (Modeling Outcomes)

### On-slide content
- Unified evaluation across model families
- Champion selection under leakage-safe test protocol
- Tuned DL improved representation quality
- ML remained strong baseline for stability/speed

### Speaker notes
- "We compare performance in one framework, not in isolated scripts."
- "The model choice is metric-informed but also constrained by serving latency and reliability."
- "This let us retain both a high-performing option and robust fallbacks."

### Decision and rationale
- **Decision:** preserve multiple deployable models (not single winner only).
- **Based on:** reliability and graceful degradation in production.
- **Outcome:** if one artifact is unavailable, app can still serve recommendations.

---

## Slide 6 - Results 2 (Deployment + Explainability)

### On-slide content
- FastAPI endpoints: health, models, predict, recommend, NLP query
- Streamlit tabs: recommend, model inspector, embedding, visualizers, evidence, AI concepts
- Diversity reranking and taste-map visualization
- Runtime NLP modes with guardrails

### Speaker notes
- "The recommend flow is served via API and rendered in Streamlit."
- "Taste Map compares user historical profile vs current recommended mix."
- "Diversity slider uses a reranking penalty to avoid repetitive genre stacks."
- "We log timings and request IDs for observability."

### Decision and rationale
- **Decision:** add diversity reranking with explainability fields.
- **Based on:** UX feedback that pure relevance can look repetitive.
- **Outcome:** controllable novelty without hiding recommendation math.

---

## Slide 7 - Conclusions

### On-slide content
- End-to-end capstone delivered from data to deployment
- Reproducibility, explainability, and maintainability achieved
- Cold-start for brand-new users/items is identified as future work (not fully implemented yet)
- Next: full local LLM integration, better cold-start onboarding, stronger evaluation loop

### Speaker notes
- "The project demonstrates full-stack AI development, not just model training."
- "Main contributions are lifecycle discipline, model interoperability, and transparent recommendation behavior."
- "Future work prioritizes local LLM integration, online-style evaluation, and presentation-layer polish."

### Decision and rationale
- **Decision:** document all phases and evidence artifacts in-app and in repo.
- **Based on:** reproducibility and handover needs.
- **Outcome:** any reviewer can trace what was run, why decisions were made, and what outputs were produced.

---

## Two-Minute Q&A Pack (High-Probability Questions)

Use concise structure: **Question -> short answer -> implementation evidence**.

### Q1) Why time-based split instead of random split?
- Random split leaks future behavior into train.
- Time-based split mirrors real-world serving: train on past, predict future.
- This gives honest generalization performance.

### Q2) How do you explain recommendations?
- API returns per-item fields like predicted score and diversity-adjusted score.
- UI shows diversity math and overlap penalties.
- Model Inspector exposes metrics/params/availability for transparency.

### Q3) Why keep multiple models instead of just one best model?
- Different models trade off latency, robustness, and explainability.
- If one artifact fails to load, fallback models keep system available.
- Useful for demos and production resilience.

### Q4) What does "normalized" mean in the Taste Map?
- Each profile (user history and recommended mix) is scaled to [0,1] independently.
- 1.0 is strongest genre inside that profile, not absolute count across users.
- Helps compare shape of preferences rather than volume.

### Q5) What was the biggest engineering challenge?
- Artifact/environment mismatch risk (e.g., model loading robustness).
- Solved by defensive registry loading + availability flags.
- Added endpoint-level health and timing logs for diagnosis.

### Q6) What is your local LLM status?
- Runtime architecture exists with guardrails and fallback parser.
- Full local provider integration (e.g., Ollama inference path) is scoped as next-step implementation.

### Q7) How is this submission reproducible?
- Structured repository and phase-wise evidence artifacts.
- Standardized run commands and docs across README/handovers/wikis.
- API and UI smoke validations captured in evidence folders.

### Q8) Did you implement cold-start handling?
- Not as a dedicated algorithmic module in this version.
- Current system has practical fallbacks (e.g., available model defaults), but no full new-user preference onboarding workflow.
- This is explicitly tracked as future work in conclusions.

---

## Decision Register (Quick Reference)

1. **Time-based split**
   - Why: leakage prevention
   - Based on: temporal recommendation nature
   - Outcome: trustworthy offline evaluation

2. **Model registry across families**
   - Why: one serving interface for Baseline/ML/DL/Tuned-DL
   - Based on: project requirement for selectable models
   - Outcome: extensible API + UI model switch

3. **Batched scoring in recommendation**
   - Why: lower response latency
   - Based on: observed slower per-item scoring
   - Outcome: faster Top-N generation

4. **Diversity reranking**
   - Why: reduce repetitive recommendations
   - Based on: user UX feedback
   - Outcome: controllable novelty (`alpha`) with visible math

5. **Model Inspector + evidence tabs**
   - Why: explainability and auditability
   - Based on: capstone reproducibility rules
   - Outcome: transparent model metadata and run artifacts

---

## Implementation-Centric Talking Points (If Asked Deep Technical Questions)

- Data path: raw -> processed features -> train/test by time.
- Serving path: Streamlit action -> FastAPI endpoint -> model registry -> scored output.
- Explainability path: metadata + metrics + visualizers + per-item recommendation fields.
- Reliability path: graceful model availability handling + middleware timings + health endpoint.
- Governance path: phase documents + evidence capture + periodic code review notes.
