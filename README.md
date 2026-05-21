# ALF Scan Viewer

Standalone viewer for ALF Surface Scanner outputs.

## Run from source

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_runtime.ps1
python -m alf_scan_viewer "C:\path\to\gridmap.npz"
```

You can also use the root runner:

```powershell
python run_viewer.py "C:\path\to\gridmap.npz"
```

You can open:

- ALF gridmaps as `.npz` files with a matching `.json` sidecar
- `.ply` geometry files
- point cloud files supported by Open3D: `.pcd`, `.pts`, `.xyz`, `.xyzn`, `.xyzrgb`

For gridmaps, the viewer reads `mean` as height and `rgb` as color, then shows
the scan as one colored 3D point set. There are no separate display modes yet.

## Open Directly From A File

The app accepts a file path as its first argument:

```powershell
python -m alf_scan_viewer "C:\path\to\surface_mesh.ply"
```

When packaged as an executable, this same argument contract is what Windows file
associations or the ALF Tool Kit launcher should use.

## Build A Windows Exe

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
