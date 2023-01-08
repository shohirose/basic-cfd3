$simulators = "simulator_sw_1st_order", "simulator_roe_1st_order", "simulator_lax_wendroff", "simulator_exact"
$buildFolders = ".\build\Release", ".\build\Debug", ".\build"
foreach ($folder in $buildFolders) {
  if (Test-Path $folder) {
    foreach ($simulator in $simulators) {
      $path = Join-Path $folder $simulator
      & $path
    }
    break
  }
}