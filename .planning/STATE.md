# STATE: ViHateT5 Improvements

## Current Phase
**Phase 4: Ensemble Evaluation** — Completed

## Last Session
- **2026-05-31**: Executed Phase 4 (Ensemble Evaluation)
  - Fixed src/ensemble.py: MPS device detection + label_remap parameter for BERT models
  - Created scripts/run_ensemble.py CLI with full argparse (--models, --all-models, --task, --no-optimize, --weights, --batch-size, --output, --data-file)
  - Ran ensemble evaluation: 3 models (vit5_finetune_balanced + vit5_focal_loss_exp + visobert_labeling)
  - Generated results/ensemble_results.csv with individual + ensemble metrics
  - Added README section (### 9. Model Ensemble) with usage, parameters, and results
  - Added 17 unit tests in tests/test_ensemble_cli.py (all pass)
  - Full suite: 166 passed, 1 skipped, 1 xfailed

## Last Action
- **2026-05-31**: Completed Phase 2 (Focal Loss Integration) — both plans executed
  - Plan 02-01: Integrated focal loss CLI flags into train_t5.py and run_train_t5.sh
  - Plan 02-02: Unit tests (12/12 pass), README docs, comparison CSV (FL macro F1=0.7478 vs CE=0.6698)

## Progress

### Phase Status
| Phase | Status | Progress |
|-------|--------|----------|
| 1. Fix Tests | `COMPLETED` | 100% |
| 2. Focal Loss | `COMPLETED` | 100% |
| 3. Augmentation | `COMPLETED` | 100% |
| 4. Ensemble | `COMPLETED` | 100% |
| 5. Error Analysis | `NOT_STARTED` | 0% |
| 6. Documentation | `NOT_STARTED` | 0% |

### Metrics
- **Tests**: 166 passing (100%) + 1 skipped + 1 xfail
- **Skipped**: 1 (requires optional underthesea package)

## Key Findings (Analysis Session)

### Code Comparison: Reference vs Current
| Aspect | Reference | Current | Delta |
|--------|-----------|---------|-------|
| src/ files | 12 | 17 | +5 new modules |
| Tests | N/A | 129 | Comprehensive |
| Shell scripts | 3 | 4 | +push_models_to_hf.sh |

### Unintegrated Modules (Critical)
1. ~~**focal_loss.py** — Implemented but NOT used in train_t5.py~~ ✅ Integrated (Phase 2)
2. **augment.py** — ~~Implemented but NOT integrated~~ ✅ Integrated (Phase 3)
3. ~~**ensemble.py** — Implemented but no execution script~~ ✅ Integrated (Phase 4)
4. **error_analysis.py** — Implemented but no CLI script

### Models on HuggingFace (All Deployed)
- NCPhat2005/vihatet5_reimpl ✅
- NCPhat2005/visobert_labeling ✅
- NCPhat2005/vit5_finetune_balanced ✅
- NCPhat2005/vit5_finetune_hate_only ✅
- NCPhat2005/vit5_finetune_multi ✅
- NCPhat2005/vit5_pretrain_balanced ✅
- NCPhat2005/vit5_pretrain_hate_only ✅

## Decisions

### DEC-001: Improvement Approach
- **Decision**: Model-centric improvements first (focal loss), then data-centric (augmentation)
- **Rationale**: Focal loss is lower risk, faster to implement
- **Date**: 2026-05-30

### DEC-002: Test Fix Priority
- **Decision**: Fix failing tests before any new features
- **Rationale**: Maintain code quality foundation
- **Date**: 2026-05-30

## Blockers
None currently.

## Next Actions
1. Analyze failing tests in `test_inference.py`
2. Fix mock model setup or test expectations
3. Verify all 129 tests pass
4. Proceed to Phase 2 (Focal Loss Integration)

---

## Session Log

### 2026-05-30 — Initial Analysis
- Reviewed project structure
- Compared with reference/code/
- Ran test suite: 124 passed, 4 failed
- Identified unintegrated modules
- Created planning structure

### 2026-05-30 — Phase 1 Complete
- Diagnosed test failures: NumPy 2.x incompatible with PyTorch 2.2.2
- Fixed: `pip install "numpy<2"` → numpy 1.26.4
- Updated requirements.txt to pin `numpy>=1.21.0,<2`
- All tests passing: 128/129 (1 skipped - optional dependency)

---
*Last updated: 2026-05-30*
