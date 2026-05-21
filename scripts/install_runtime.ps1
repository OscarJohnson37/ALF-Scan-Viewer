$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $root "requirements.txt"

python -m pip install -r $requirements

# Open3D's wheel pulls in Jupyter widget assets that can exceed Windows' default
# path length limit. The viewer only needs the desktop Open3D runtime, so install
# the wheel after the small runtime deps are present.
python -m pip install open3d==0.19.0 --no-deps

python -c "import open3d as o3d; import open3d.visualization.gui; print('Open3D', o3d.__version__)"

