# ROADMAP: ViHateT5 Improvements

## Milestone: v2.0 - Model-Centric Improvements

### Overview
Tích hợp các cải tiến model-centric đã implement vào training pipeline và evaluation.

---

## Phase 1: Fix Tests & Code Quality
**Status**: `COMPLETED`
**Priority**: HIGH
**Estimated**: 1-2 hours
**Requirements:** [IMP-01]

### Goals
- Fix 4 failing tests in `test_inference.py`
- Ensure 100% test pass rate (or 128/129 with 1 correctly skipped)

### Plans: 1 plan

Plans:
- [x] 01-01-PLAN.md — Diagnose NumPy 2.x incompatibility and pin numpy<2

### Success Criteria
- [x] 128/129 tests pass (1 correctly skipped for optional dependency)
- [x] NumPy pinned to <2 in requirements.txt

---

## Phase 2: Focal Loss Integration
**Status**: `COMPLETED`
**Priority**: HIGH
**Estimated**: 2-3 hours
**Requirements:** [IMP-02]

### Goals
- Integrate focal loss vào T5 training pipeline
- Add CLI flags for configuration
- Document performance impact

### Plans: 2 plans

Plans:
- [x] 02-01-PLAN.md — Core integration: CLI args, trainer selection, shell script
- [x] 02-02-PLAN.md — Tests, documentation, and experiments with comparison table

### Tasks
| ID | Task | File | Complexity |
|----|------|------|------------|
| 2.1 | Add focal loss imports | `src/train_t5.py` | Low |
| 2.2 | Add CLI arguments | `src/train_t5.py` | Low |
| 2.3 | Replace Seq2SeqTrainer conditionally | `src/train_t5.py` | Medium |
| 2.4 | Update shell script | `scripts/run_train_t5.sh` | Low |
| 2.5 | Add tests for focal loss | `tests/test_focal_loss.py` | Medium |
| 2.6 | Run experiment and document | README.md | Medium |

### Success Criteria
- [x] `--use_focal_loss` flag works
- [x] `--focal_gamma` configurable
- [x] `--label_smoothing` optional
- [x] Training completes without errors
- [x] Comparison table: CE vs Focal Loss

---

## Phase 3: Data Augmentation Pipeline
**Status**: `PLANNED`
**Priority**: MEDIUM
**Estimated**: 3-4 hours
**Requirements:** [IMP-03]

### Goals
- Integrate augmentation vào data loading
- Support selective augmentation cho minority classes

### Plans: 1 plan

Plans:
- [ ] 03-01-PLAN.md — CLI integration, augmentation pipeline, tests, standalone script

### Tasks
| ID | Task | File | Complexity |
|----|------|------|------------|
| 3.1 | Add augmentation to DataLoader | `src/data_loader.py` | Medium |
| 3.2 | Add CLI flags | `src/train_t5.py`, `src/train_bert.py` | Low |
| 3.3 | Create augmentation config | `src/config.py` | Low |
| 3.4 | Selective class augmentation | `src/augment.py` | Medium |
| 3.5 | Create standalone script | `scripts/run_augment.py` | Medium |
| 3.6 | Add tests | `tests/test_augment.py` | Medium |

### Success Criteria
- [ ] `--augment_minority` flag works
- [ ] `--augment_factor N` configurable
- [ ] HATE class augmented x3 by default
- [ ] Training with augmentation completes

---

## Phase 4: Ensemble Evaluation
**Status**: `PLANNED`
**Priority**: MEDIUM
**Estimated**: 2-3 hours
**Requirements:** [REQ-04]

### Goals
- Create script để chạy ensemble từ multiple models
- Support different voting strategies

### Plans: 1 plan

Plans:
- [ ] 04-01-PLAN.md — Ensemble CLI script, MPS/label_remap fix, results CSV, README, tests

### Tasks
| ID | Task | File | Complexity |
|----|------|------|------------|
| 4.1 | Create ensemble CLI script | `scripts/run_ensemble.py` | Medium |
| 4.2 | Support T5 + BERT combination | `scripts/run_ensemble.py` | Medium |
| 4.3 | Implement voting strategies | `src/ensemble.py` (extend) | Low |
| 4.4 | Output comparison metrics | - | Low |
| 4.5 | Add to README | `README.md` | Low |

### Success Criteria
- [ ] Script runs with model paths
- [ ] Outputs ensemble F1, accuracy
- [ ] Comparison: single vs ensemble

---

## Phase 5: Error Analysis & Visualization
**Status**: `COMPLETED`
**Priority**: MEDIUM
**Estimated**: 2-3 hours
**Requirements:** [REQ-03]

### Goals
- Comprehensive error analysis of all 5 fine-tuned models with confusion matrices, per-class F1 breakdowns, misclassification analysis
- McNemar's statistical significance tests (star pattern + ensemble comparison)
- Combined comparison chart integrating Phase 2/4 results

### Plans: 2 plans

Plans:
- [x] 05-01-PLAN.md — Add McNemar's test, combined comparison chart, per-model filename support to src/error_analysis.py
- [x] 05-02-PLAN.md — Rewrite CLI for multi-model analysis with BERT support, prediction caching, McNemar, combined chart

### Success Criteria
- [x] Confusion matrices generated for all 5 models
- [x] Per-class F1 bar charts per model
- [x] McNemar significance test results (star pattern + ensemble)
- [x] Combined comparison chart (all models + ensemble)
- [x] All saved to `results/images/` and `results/analysis/`

---

## Phase 6: Documentation & Demo
**Status**: `COMPLETED`
**Priority**: LOW
**Estimated**: 3-4 hours

### Goals
- Update README với all new features (focal loss, augmentation, ensemble, error analysis)
- Update Streamlit app (`app.py`) với improvement results (Phase 2-5), ensemble demo, error analysis visualizations
- Update FastAPI webapp (`webapp/`) để hiển thị kết quả mới: ensemble, McNemar, combined comparison chart
- Create demo notebook

### Plans: 1 plan

Plans:
- [x] 06-01-PLAN.md — README sections 10-11, Streamlit Improvements tab, FastAPI /results route, improvements notebook

### Tasks
| ID | Task | File | Complexity |
|----|------|------|------------|
| 6.1 | Update Usage section | `README.md` | Low |
| 6.2 | Add experiment tables (focal loss, ensemble, error analysis) | `README.md` | Medium |
| 6.3 | Add improvement results tab to Streamlit app | `app.py` | Medium |
| 6.4 | Add ensemble demo to Streamlit app | `app.py` | Medium |
| 6.5 | Add error analysis visualizations to Streamlit app | `app.py` | Medium |
| 6.6 | Update FastAPI webapp with improvement results page | `webapp/main.py`, `webapp/templates/` | Medium |
| 6.7 | Add ensemble + McNemar results to FastAPI webapp | `webapp/main.py`, `webapp/templates/` | Medium |
| 6.8 | Create improvements notebook | `notebooks/improvements.ipynb` | Medium |
| 6.9 | Update Limitations section | `README.md` | Low |

### Success Criteria
- [x] README documents all features (focal loss, augmentation, ensemble, error analysis)
- [x] Streamlit app shows Phase 2-5 improvement results and comparison charts
- [x] Streamlit app supports ensemble inference demo
- [x] FastAPI webapp displays improvement results, ensemble, and error analysis
- [x] Notebook runs end-to-end
- [x] Experiment results visible in all interfaces

---

## Timeline Summary

```
Phase 1 (Fix Tests)         ████░░░░░░░░░░░░░░░░  1-2h
Phase 2 (Focal Loss)        ████████░░░░░░░░░░░░  2-3h
Phase 3 (Augmentation)      ████████████░░░░░░░░  3-4h
Phase 4 (Ensemble)          ████████░░░░░░░░░░░░  2-3h
Phase 5 (Error Analysis)    ████████░░░░░░░░░░░░  2-3h
Phase 6 (Documentation)     ████████░░░░░░░░░░░░  2-3h
                            ────────────────────
                            Total: 12-18 hours
```

---

## Dependencies Between Phases

```
Phase 1 (Tests) ──┬──> Phase 2 (Focal)
                  │
                  ├──> Phase 3 (Augment)
                  │
                  └──> Phase 4 (Ensemble) ──> Phase 5 (Analysis)
                                                      │
                                                      v
                                              Phase 6 (Docs)
```

- Phase 1 should be done first (foundation)
- Phases 2, 3, 4 can run in parallel
- Phase 5 requires models from Phase 4
- Phase 6 requires all results

---
*Generated: 2026-05-30*
