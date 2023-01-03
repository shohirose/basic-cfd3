$simulators = "simulator_sw_1st_order", "simulator_roe_1st_order", "simulator_exact"
foreach ($simulator in $simulators) {
  & .\build\$simulator
}