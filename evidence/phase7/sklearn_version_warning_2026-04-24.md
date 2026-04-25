# sklearn Version Warning Note (2026-04-24)

During API test execution, sklearn emitted compatibility warnings while loading persisted `.pkl` artifacts.

Observed warning type:
- `InconsistentVersionWarning` (artifacts trained with sklearn 1.6.1, runtime using 1.8.0)

Current mitigation:
- API registry avoids strict parameter introspection that can break on version mismatch.
- API still reports model availability and provides inference paths.

Recommended long-term fix:
- retrain and persist ML artifacts using the pinned production/runtime sklearn version.

