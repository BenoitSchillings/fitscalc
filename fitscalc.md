# fitscalc — interactive FITS image calculator

A small REPL for combining and processing FITS images with arithmetic
expressions. Bare names refer to FITS files in the current directory, so
`a = b + c` reads `b.fits` and `c.fits`, adds them per-pixel, and writes
the result to `a.fits`.

## Running

```
python3 fitscalc.py                    # start the REPL
python3 fitscalc.py 'a = b + c'        # one-shot expression
python3 fitscalc.py < script.txt       # batch mode (one statement per line)
```

Inside the REPL: line editing and history are enabled. Lines starting
with `#` are comments. Type `quit` (or `exit`, `q`, or Ctrl-D) to leave.

## Mental model

- A bare identifier `x` means the file `x.fits` in the current
  directory. There are no in-memory variables — files *are* the state.
- An assignment `lhs = expr` evaluates `expr` and writes `lhs.fits`.
- An expression on its own line prints summary statistics.
- A scalar result on the right of an assignment is **printed**, not
  written to disk (so `n = count(lights*)` won't litter your folder
  with `n.fits`).

## Expressions

### Operators

| Op  | Meaning              | Example         |
| --- | -------------------- | --------------- |
| `+` | add                  | `a + b`, `a + 1`|
| `-` | subtract / unary neg | `a - b`, `-a`   |
| `*` | multiply             | `a * 0.5`       |
| `/` | divide               | `a / b`         |
| `**`| power                | `a ** 2`        |
| `%` | modulo               | `a % 1`         |
| `<`, `<=`, `>`, `>=`, `==`, `!=` | element-wise compare | `a > 100` |

Numeric literals work normally: `b * 0.5`, `b - 1000`, `a ** 0.5`.

### File references

There are three forms:

```
bare name              b               -> b.fits
bare glob              lights*         -> all lights*.fits, as a list
quoted path/glob       "darks/*.fits"  -> all matching files, as a list
```

A *list* of images is what you pass to a stack reduction
(`mean`, `median`, `mean_sigma`, …). Assigning a list to a name is an
error — wrap it in a reduction first.

### Glob disambiguation

Globs are recognized only in argument-list position (after `(`, `,`, or
`=`). So:

```
mean(b*c)        # glob: matches b*c.fits — no files? error
mean(b * c)      # multiplication: per-pixel product
b * 2            # multiplication
a = b * 2        # multiplication (RHS, but not adjacent to `=`)
oops = b*c       # treated as glob
```

When in doubt, put spaces around `*` for arithmetic, no spaces for globs.

## Function reference

### Element-wise math

```
sqrt(x)    log(x)    log10(x)   log2(x)
exp(x)     abs(x)    clip(x, lo, hi)
where(cond, a, b)
```

`where` is element-wise pick: `where(b > 1000, 0, b)` zeroes out pixels
above 1000.

### Stack reductions (per-pixel across multiple images)

```
mean(images*)            median(images*)
sum(images*)             std(images*)
min(images*)             max(images*)
```

Each accepts:

- a glob:           `mean(lights*)`
- multiple names:   `mean(l1, l2, l3)`
- a quoted glob:    `mean("data/lights*.fits")`

When given a single image they reduce all axes (return a scalar), so
`mean(b)` is the overall mean of `b.fits`.

### Sigma-clipped stack reductions

```
mean_sigma(images*, k=3, maxiters=5)
median_sigma(images*, k=3, maxiters=5)
```

Iteratively reject pixels deviating more than `k * MAD` from the median,
then reduce. The robust default (`cenfunc=median`, `stdfunc=mad_std`)
prevents the rejection threshold from being inflated by the very
outliers it is trying to clip — use these instead of plain `mean` for
real stacking, where a single cosmic-ray hit ruins the per-pixel mean.

### Single-image utilities

```
bg(img, k=3, maxiters=5)              # sigma-clipped sky-background scalar
bg2d(img, box=64, filter_size=3)      # 2D background map (image-shaped)
percentile(img, p)                    # e.g. percentile(b, 99.5)
crop(img, x1, y1, x2, y2)             # end-exclusive: matches img[y1:y2, x1:x2]
```

`crop`'s coordinates are 0-based and end-exclusive — i.e. `crop(b, 0, 0, 100, 100)`
is a 100×100 sub-image. `bg` returns the sky-background level (a scalar)
robust to stars; subtract it with `flat = b - bg(b)`.

`bg2d` fits a smooth surface to the background using meshes of size
`box × box` pixels (medianed, then median-filtered with `filter_size`),
returning an *image* the same shape as the input. Use it instead of `bg`
when the sky is non-uniform — light-pollution gradients, vignetting,
amplifier glow — where one number can't capture the structure.

```
fits> flat = b - bg2d(b, box=64)
```

A larger `box` gives a smoother (more conservative) background; smaller
`box` follows finer structure but risks absorbing extended sources.

### Counting files

```
count(lights*)
```

Returns the number of files matching the glob (without loading them).

### Viewing images

```
view(b)                # open b.fits in the image viewer
view(lights*)          # browse a sequence; use Prev/Next or arrow keys
view(b, c, master_dark)
```

`view()` launches the GUI viewer (`view.py`) as a separate process — the
REPL stays usable while the viewer is open. Only file references are
allowed (bare names, globs, quoted paths); to inspect a computed image,
assign it first and view the resulting file:

```
fits> stack = mean_sigma(lights*, k=3)
fits> view(stack)
```

A second `view()` call reuses the existing window rather than opening a
new one. Communication goes over a Unix-domain socket at
`/tmp/fitscalc-viewer-<uid>.sock`; if the viewer is closed (or crashed),
the next `view()` call spawns a fresh instance. Each `view()` *replaces*
the navigator's file list — there is no append mode.

```
fits> view(lights*)        # open and browse the lights sequence
fits> view(stack)          # same window now shows stack.fits
```

Useful viewer shortcuts: `A` auto-scale to median..99th-percentile,
`F` toggle fullscreen, `n` print the current filename, `d` delete the
current file. Click on a star to print its HFD and FWHM in the terminal.

The viewer location can be overridden with the `FITSCALC_VIEWER`
environment variable.

**`view.py` runtime dependencies.** The viewer needs PyQt5, pyqtgraph,
opencv-python, scipy, and numba (the last for `util.py`'s outlier
filter). `util.py` and `ser.py` are bundled in the repo; they live next
to `view.py` and are imported directly. Set `FITSCALC_VIEWER` to point
at a different viewer if you have your own.

## Special commands

```
ls                  list *.fits in the current directory
info <name>         show FITS HDU info for <name>.fits
stats <name>        print min/max/mean/std of <name>.fits
help                show the docstring
quit | exit | q     leave
```

(`view` is a function-call form, not a special command — see
"Viewing images" above.)

## Examples

### Inspection

```
fits> ls
fits> stats raw
fits> info raw
fits> raw                        # bare expression -> stats
fits> bg(raw)
fits> percentile(raw, 99.9)
fits> count(lights*)
```

### Master calibration frames

Build masters from a sequence of calibration exposures:

```
fits> master_bias = median_sigma(bias*, k=3)
fits> master_dark = median_sigma(dark*, k=3) - master_bias
fits> master_flat_raw = median_sigma(flat*, k=3) - master_bias
fits> master_flat = master_flat_raw / percentile(master_flat_raw, 50)
```

The flat is normalized by its own median so dividing by it preserves
overall flux scale.

### Calibrated light frame

```
fits> cal01 = (light01 - master_dark) / master_flat
```

### Stacking calibrated lights

If you have many calibrated lights, stack with sigma rejection:

```
fits> final = mean_sigma(cal*, k=3)
```

Compare to a plain mean to see how much the rejection helps:

```
fits> dirty = mean(cal*)
fits> diff  = final - dirty
fits> stats diff
```

`diff` will be near zero almost everywhere, with localized differences
where outliers (cosmic rays, satellite trails) were removed.

### Background subtraction

Scalar (uniform sky):

```
fits> flat = final - bg(final)
```

2D background map (light-pollution gradient or vignetting):

```
fits> back = bg2d(final, box=128)
fits> flat = final - back
fits> stats flat
```

Inspect the background itself — useful for verifying it didn't absorb
your target:

```
fits> back
fits> diff = final - back
```

Per-percentile pedestal (occasionally useful for display):

```
fits> display = final - percentile(final, 25)
```

### Region of interest

Crop a tile and process it independently:

```
fits> nebula = crop(final, 1200, 800, 2400, 1800)
fits> stats nebula
```

`crop` arguments are `(image, x1, y1, x2, y2)` end-exclusive, so this
yields a 1200×1000 sub-image (`x: 1200..2399`, `y: 800..1799`).

### Difference imaging

Subtract a reference frame to highlight changes (transients, comet
motion, variable stars):

```
fits> ref = median_sigma(reference_lights*, k=3)
fits> diff = current - ref
fits> stats diff
fits> hot  = where(diff > 5 * std(diff), diff, 0)
```

### Normalization to common flux

Useful before averaging frames taken under varying conditions:

```
fits> n01 = light01 / percentile(light01, 50)
fits> n02 = light02 / percentile(light02, 50)
fits> n03 = light03 / percentile(light03, 50)
fits> avg = mean_sigma(n01, n02, n03, k=3)
```

A glob shortcut works once you've named them consistently:

```
fits> avg = mean_sigma(n*, k=3)
```

### Mask-based pixel replacement

Use comparisons to build a mask, then `where` to substitute:

```
fits> hot_mask = light01 > 60000
fits> repaired = where(hot_mask, master_dark, light01)
```

(Comparisons return float arrays of 0/1 here, which combine naturally
with `where` and arithmetic.)

### Photometry-flavored quick checks

```
fits> sky      = bg(light01)
fits> bright   = percentile(light01, 99.9)
fits> contrast = bright - sky
```

### One-shot batch processing

Save a routine pipeline to a file:

```
# pipeline.txt
master_bias = median_sigma(bias*, k=3)
master_dark = median_sigma(dark*, k=3) - master_bias
master_flat = median_sigma(flat*, k=3) - master_bias
master_flat = master_flat / percentile(master_flat, 50)
cal01 = (light01 - master_dark) / master_flat
cal02 = (light02 - master_dark) / master_flat
cal03 = (light03 - master_dark) / master_flat
final = mean_sigma(cal*, k=3)
flat  = final - bg(final)
```

Run it:

```
python3 fitscalc.py < pipeline.txt
```

## Caveats

- **Output is float32.** Inputs are read as float64 for arithmetic,
  results are saved as float32.
- **Headers are not preserved.** WCS, exposure time, filter, etc. from
  the input HDUs are dropped on write. (Easy to add — ask.)
- **Crop is 0-based and end-exclusive.** This matches NumPy slicing,
  not 1-based FITS pixel coordinates.
- **No registration / alignment.** All multi-image operations assume
  the inputs are pixel-aligned.
- **Stack reductions need ≥ 2 frames.** Sigma-clipped reductions also
  need a glob or multiple names — they error on a single image.
- **No in-memory variables.** Every expression that produces an image
  goes to disk. Use shorter names for intermediate products if this
  bothers you, or wrap a long pipeline in batch mode.
- **Glob characters in arithmetic.** `b*c` (no spaces) is treated as a
  glob. Use `b * c` for multiplication.
