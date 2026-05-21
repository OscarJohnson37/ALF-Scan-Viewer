$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "dist"
$work = Join-Path $root "build"
$spec = Join-Path $root "build_spec"

& (Join-Path $root "scripts\install_runtime.ps1")
python -m pip install pyinstaller

python -m PyInstaller --noconfirm --clean --windowed --onefile `
    --name "ALF Scan Viewer" `
    --distpath $dist `
    --workpath $work `
    --specpath $spec `
    --collect-all open3d `
    (Join-Path $root "run_viewer.py")

Write-Host "Built viewer:" (Join-Path $dist "ALF Scan Viewer.exe")
