param(
    [string]$ViewerExe = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if (-not $ViewerExe) {
    $defaultExe = Join-Path $root "dist\ALF Scan Viewer.exe"
    if (Test-Path $defaultExe) {
        $ViewerExe = $defaultExe
    } else {
        $python = (Get-Command python -ErrorAction Stop).Source
        $pythonw = Join-Path (Split-Path -Parent $python) "pythonw.exe"
        if (Test-Path $pythonw) {
            $python = $pythonw
        }
        $runner = Join-Path $root "run_viewer.py"
        if (-not (Test-Path $runner)) {
            throw "Source runner not found: $runner"
        }
        $command = '"' + $python + '" "' + $runner + '" "%1"'
    }
}

if ($ViewerExe) {
    $ViewerExe = (Resolve-Path $ViewerExe).Path
    $command = '"' + $ViewerExe + '" "%1"'
}

$progId = "ALFScanViewer.File"
$extensions = @(
    ".npz",
    ".ply",
    ".pcd",
    ".pts",
    ".xyz",
    ".xyzn",
    ".xyzrgb"
)

New-Item -Force "HKCU:\Software\Classes\$progId" | Out-Null
Set-Item "HKCU:\Software\Classes\$progId" -Value "ALF Scan Viewer File"
New-Item -Force "HKCU:\Software\Classes\$progId\shell\open\command" | Out-Null
Set-Item "HKCU:\Software\Classes\$progId\shell\open\command" -Value $command

foreach ($extension in $extensions) {
    New-Item -Force "HKCU:\Software\Classes\$extension" | Out-Null
    Set-Item "HKCU:\Software\Classes\$extension" -Value $progId
}

Write-Host "Registered ALF Scan Viewer for:" ($extensions -join ", ")
Write-Host "Command:" $command
