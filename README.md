# ALF Scan Viewer

Standalone viewer for ALF Surface Scanner outputs.

## Run from source

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_runtime.ps1
python -m alf_scan_viewer "C:\path\to\ALF Project"
```

You can also use the root runner:

```powershell
python run_viewer.py "C:\path\to\ALF Project"
```

You can open either:

- an ALF project folder
- the project's `data\tools\surface_scanner` folder

The viewer lists Surface Scanner milestones from that folder. When a raw
`.npz` gridmap exists, the viewer renders from that so the live controls apply.
Processed `*_surface_mesh.ply` files are used as a fallback when no raw gridmap
is available.

The side panel uses expandable sections:

- `Project`: choose selected/stacked view, pick a milestone, or manually refresh
- `Surface`: choose scanner RGB, height, signed deformation, or absolute
  deformation colouring, then choose the scalar source such as mean, median,
  variance, visual-smoothed mean, or the processed weighted spline mean and
  residual when that gridmap exists; optionally switch supported scalar
  colouring to binary
  red/green with a threshold in millimetres and invert toggle; signed
  deformation gradients can also be inverted to choose whether up or down is
  red; set downsampling
- `Z Axis`: choose height, signed deformation, absolute deformation, or flat,
  then choose the scalar source used by that axis
- `Section`: optionally limit rendering to an X and/or Y coordinate range
- `Stack`: choose which raw-gridmap milestones are included and set the
  vertical offset between them in stacked view
- `Camera`: rotate with the mouse, apply a preset view, and export the current
  camera view as a Surface Scanner graph PNG
- `Info`: show the loaded files, baseline, and stack order

Changing these controls refreshes the view automatically. Stack separation uses
the cached stack, so dragging it should feel continuous.

Exported camera views are written to:

```text
data\tools\surface_scanner\graphs\3d_surface_views
```

The export uses the current rendered Open3D camera, so a manually rotated view is
saved exactly as shown.

The baseline dropdown defaults to the ALF shared milestone marked with
`post_bedding_baseline` when that metadata exists. Boundary/material marker
lines are possible in the Open3D scene, but need a project metadata source or
manual boundary editor before they can be drawn reliably.

The `Mean (visual smoothed)` source is a display layer, not corrected
measurement data. It builds visual confidence from the mean cell's `variance`
and point `count`, uses a local robust median/MAD check to identify uncertain
spikes, and repairs only those visual outliers from trusted neighbours. Raw
`Mean` stays untouched for real analysis. If visual smoothing is used for
deformation, that same confidence-aware outlier pass is applied to the
deformation view so the comparison is not made from two independently smoothed
height maps.

## Open Directly From A File

The app accepts a project or Surface Scanner folder as its first argument:

```powershell
python -m alf_scan_viewer "C:\path\to\Project\data\tools\surface_scanner"
```

Direct file opening still works as a fallback for `.npz`, `.ply`, `.pcd`,
`.pts`, `.xyz`, `.xyzn`, and `.xyzrgb`, but the main workflow is milestone
selection from a project directory.

## Build A Windows Exe

Put the app icon at:

```text
assets\ALF_Scan_Viewer.ico
```

Use an `.ico` file for the Windows executable icon. If your source artwork is a
`.jpg`, convert/export it to `ALF_Scan_Viewer.ico` first. The build script passes
that icon to PyInstaller, which gives the exe its icon and is also what Windows
uses for the app window icon.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

The output is:

```text
dist\ALF Scan Viewer.exe
```

## Register File Associations

After building the exe, register supported file types for the current Windows
user:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_file_associations.ps1
```

Or point the script at an exe manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_file_associations.ps1 -ViewerExe "C:\path\to\ALF Scan Viewer.exe"
```

After that, opening a supported file from Explorer should launch ALF Scan Viewer
with that file path.

If the packaged exe has not been built yet, the registration script falls back
to launching the source checkout through `pythonw.exe`.

The app does not register itself as a `.json` opener. Gridmap JSON files are
treated as sidecars for matching `.npz` files.
