# Draft: Autodesk/XLB

**Status:** draft, not filed. File only after Dion approves. May not be needed (see notes).

**Title:** Release request: xlb 0.3.1 fails to import with warp-lang ≥ 1.13

## Body

`import xlb` fails for a fresh `pip install xlb==0.3.1`, because pip resolves the newest warp-lang that satisfies `warp-lang>=1.10.0`. `xlb/operator/stepper/__init__.py` imports `IBMStepper` unconditionally, and `ibm_stepper.py` does `from warp.utils import ScopedTimer`. Warp 1.10 to 1.12 still resolve that name with a deprecation warning, but 1.13.0 removed it from `warp.utils`.

`main` already imports `from warp import ScopedTimer`, so the fix exists. Would you consider a 0.3.2 release with it? The metadata of 0.3.1 can't gain an upper bound after the fact, so a release seems to be the only way for pip users to get a working install without pinning Warp themselves.

Workaround for others: pin `warp-lang==1.12.1` next to `xlb==0.3.1`.

## Notes for Dion (not part of the issue)

- Related closed issue: Autodesk/XLB#157 ("unable to run example"), which showed only the deprecation warning on 1.11.1.
- Mosaic's build-time source patch can be replaced by the pin (branch `fix/xlb-pin-warp-instead-of-patch`). The fix on `main` is unreleased, so under rule 1 the pin is the workaround until a release exists. A pin is not an intervention on XLB's source at all, so filing this issue is optional.
