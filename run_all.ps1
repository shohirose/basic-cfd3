Param(
  [switch]$Release,
  [switch]$Debug,
  [string]$buildDir = ".\build"
)

function Invoke-Simulators ([Parameter(Mandatory, ValueFromPipeline)]$dir) {
  if (!(Test-Path $dir)) { 
    Write-Warning "Folder not found: ${dir}"
    return
  }
  $simulators = "simulator_sw_1st_order", "simulator_roe_1st_order", `
                "simulator_lax_wendroff", "simulator_exact", `
                "simulator_roe_tvd_minmod", "simulator_roe_tvd_vanleer", `
                "simulator_sw_tvd_minmod", "simulator_sw_tvd_vanleer"
  foreach ($simulator in $simulators) {
    $path = Join-Path $dir "${simulator}.exe"
    if (Test-Path $path) {
      & $path
    }
    else {
      Write-Warning "Path not found: ${path}"
    }
  }
}

if ($Release) {
  Join-Path $buildDir "Release" | Invoke-Simulators
}
elseif ($Debug) {
  Join-Path $buildDir "Debug" | Invoke-Simulators
}
else {
  Invoke-Simulators $buildDir
}