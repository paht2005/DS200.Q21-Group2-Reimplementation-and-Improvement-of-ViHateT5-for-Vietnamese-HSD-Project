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
**Status**: `IN_PROGRESS`
**Priority**: HIGH
**Estimated**: 2-3 hours
**Requirements:** [IMP-02]

### Goals
- Integrate focal loss vào T5 training pipeline
- Add CLI flags for configuration
- Document performance impact

### Plans: 2 plans

Plans:
- [ ] 02-01-PLAN.md — Core integration: CLI args, trainer selection, shell script
- [ ] 02-02-PLAN.md — Tests, documentation, and experiments with comparison table

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
- [ ] `--use_focal_loss` flag works
- [ ] `--focal_gamma` configurable
- [ ] `--label_smoothing` optional
- [ ] Training completes without errors
- [ ] Comparison table: CE vs Focal Loss

---

## Phase 3: Data Augmentation Pipeline
**Status**: `NOT_STARTED`
**Priority**: MEDIUM
**Estimated**: 3-4 hours

### Goals
- Integrate augmentation vào data loading
- Support selective augmentation cho minority classes

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
**Status**: `NOT_STARTED`
**Priority**: MEDIUM
**Estimated**: 2-3 hours

### Goals
- Create script để chạy ensemble từ multiple models
- Support different voting strategies

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
**Status**: `NOT_STARTED`
**Priority**: MEDIUM
**Estimated**: 2-3 hours

### Goals
- Comprehensive error analysis với visualizations
- Statistical significance testing

### Tasks
| ID | Task | File | Complexity |
|----|------|------|------------|
| 5.1 | Create analysis CLI | `scripts/run_error_analysis.py` | Medium |
| 5.2 | Generate confusion matrices | `src/error_analysis.py` | Low |
| 5.3 | Per-class breakdown charts | `src/error_analysis.py` | Medium |
| 5.4 | Add McNemar test | `src/error_analysis.py` | Medium |
| 5.5 | Save to results/images/ | - | Low |

### Success Criteria
- [ ] Confusion matrices generated
- [ ] Per-class F1 bar charts
- [ ] Statistical test results
- [ ] All saved to `results/images/`

---

## Phase 6: Documentation & Demo
**Status**: `NOT_STARTED`
**Priority**: LOW
**Estimated**: 2-3 hours

### Goals
- Update README với all new features
- Create demo notebook

### Tasks
| ID | Task | File | Complexity |
|----|------|------|------------|
| 6.1 | Update Usage section | `README.md` | Low |
| 6.2 | Add experiment tables | `README.md` | Medium |
| 6.3 | Create improvements notebook | `notebooks/improvements.ipynb` | Medium |
| 6.4 | Update Limitations section | `README.md` | Low |

### Success Criteria
- [ ] README documents all features
- [ ] Notebook runs end-to-end
- [ ] Experiment results visible

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
