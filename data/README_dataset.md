# Timeseries of Energy Systems Integration Facility (ESIF) Data Center Power Usage Effectiveness (PUE) 

Data provided in Parquet and compressed CSV formats

### Power Metrics Timeseries Fields

- `ts`:  Timestamp
- `cooling_kw`:  Cooling (kilowatts) - Captures the power used by fans and pipe trace heaters associated with outdoor cooling equipment. The dedicated tower filter pump power is also captured as cooling load.
- `energy_reuse`:  Energy Reuse Effectiveness
- `hvac_kw`:  Heating, ventilation, and air conditioning (kilowatts) - Captures fan walls, fan coils that support the data center electrical rooms, and the make-up air unit.
- `it_power_kw`:  IT equipment (kilowatts) - Captures power used by the IT equipment on the data center floor.
- `plug_and_light_kw`:  Lights and utility plugs (kilowatts) - Captures power associated with the data center and dedicated mechanical room. The crank-case heater for the emergency standby generator is also captured as light and plug load.
- `pue`:  Power Usage Effectiveness
- `pump_kw`:  Pumps (kilowatts) - Captures power from pumps that move water in the data center Energy Recover Water loop and the Tower Water loops, and also captures power used by the boost pumps that circulate water through the fan walls. Note: The tower filter pump runs constantly to filter water from the data center cooling tower system, so 2.67 kilowatts are attributed to this pump and that is not reflected in this data field.
- `day`:  Day of month

### Outside Weather Station Timeseries Fields

- `ts`:  Timestamp
- `outside_air_humidity`:  Outside air humidity - Relative humidity percent
- `outside_air_temp`:  Outside air temperature - Degrees Fahrenheit
- `day`:  Day of month

###  More Detail

- [High-Performance Computing Data Center Power Usage Effectiveness](https://www.nlr.gov/computational-science/measuring-efficiency-pue)