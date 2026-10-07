# Warp numerical checks

Run these focused checks inside the solver image with a GPU:

```sh
python tests/check_fft.py --api /tesseract/tesseract_api.py
python tests/check_projection.py --api /tesseract/tesseract_api.py --out /tmp/warp-projection-check.json
```

The FFT check compares transforms and pressure VJPs against independent NumPy
calculations. The projection check verifies the centered discrete operator,
DC/Nyquist null modes, and inviscid kinetic-energy balance in 3D.

Long forced-flow campaign scripts, raw measurements, and plots are archived
with [PR #201](https://github.com/pasteurlabs/mosaic/pull/201). They are kept
outside the adapter source. The benchmark CI exercises existing benchmark cases.
