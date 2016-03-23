# LCLS
A spectroscopy and x-ray powder diffraction analysis package for the MEC endstation at the LCLS.

<installation>

<Logging spreadsheet format>

mecana.py usage:
usage: mecana.py [-h] [--noplot] [--testing]
                 {init,spectrum,xrd,histogram,datashow,eventframes,query,showderived}
                 ...

positional arguments:
  {init,spectrum,xrd,histogram,datashow,eventframes,query,showderived}
                        sub-command help
    init                Initialize config.py in local directory.
    spectrum            Process area detector data into spectra.
    xrd                 Process quad CSPAD data into powder patterns.
    histogram           For a given dataset and detector ID, evaluate a
                        function (defined in config.py) over all events and
                        plot a histogram of the resulting values.
    datashow            For a given dataset and area detector ID, show the
                        summed detector image and save it to a file in the
                        working directory. Any detector masks specified in
                        config.py can optionally be applied.
    eventframes         For a given dataset and area detector ID, save an
                        image from each event in the directory
                        eventdata/<dataset label>/.
    query               Query a dataset based on (one or both of) logbook
                        attribute values and a user-defined event filter
    showderived         output the names of all existing derived datasets.

optional arguments:
  -h, --help            show this help message and exit
  --noplot, -n          If selected, plotting is suppressed
  --testing, -t         If selected, process only 1 out of 10 events
  
Subcommand usage:
mecana.py spectrum:
  positional arguments:
  detid                 Detector ID.
  labels                Labels of datasets to process into spectra.

optional arguments:
  -h, --help            show this help message and exit
  --events EVENTS [EVENTS ...], -ne EVENTS [EVENTS ...]
                        Events numbers for which to plot spectra. This option
                        requires LABELS to correspond to a single run
  --pxwidth PXWIDTH, -p PXWIDTH
                        Pixel width of CSPAD subregion to sum.
  --rotate, -r          Toggles the area detector's axis of integration for
                        generating spectra
  --calibration CALIBRATION, -c CALIBRATION
                        Label of dataset to use for calibration of the energy
                        scale (if --energy_ref1_energy_ref2_calibration is
                        selected but a calibration file is not provided). If
                        not provided this parameter defaults to the first
                        dataset in labels.
  --subtraction SUBTRACTION, -d SUBTRACTION
                        Label of dataset to use as a dark frame subtraction
  --energy_ref1_energy_ref2_calibration, -k
                        Enable automatic generation of energy calibration
                        based on k alpha and k beta peak locations if
                        --calibration_load_path is not given.
  --eltname ELTNAME, -e ELTNAME
                        Element name. This parameter is required for XES-based
                        calibration; i.e., for generating an energy scale
                        using ENERGY_REF1_ENERGY_REF2_CALIBRATION.
  --calibration_save_path CALIBRATION_SAVE_PATH, -s CALIBRATION_SAVE_PATH
                        Path to which to save energy calibration data if
                        calibration_load_path is unspecified and
                        --energy_ref1_energy_ref2_calibration is selected.
  --calibration_load_path CALIBRATION_LOAD_PATH, -l CALIBRATION_LOAD_PATH
                        Path from which to load energy calibration data.
  --normalization, -n   If selected, normalization is suppressed
  --nosubtraction, -ns  If selected, background subtraction is suppressed
  --variation, -v       Plot shot to shot variation
  --variation_n VARIATION_N, -vn VARIATION_N
                        How many shots to use for variation
  --variation_center VARIATION_CENTER, -vc VARIATION_CENTER
                        to the right is the pump, to the left is the probe
  --variation_skip_width VARIATION_SKIP_WIDTH, -vs VARIATION_SKIP_WIDTH
                        from vc skip this many inds before summing pump and
                        probe area
  --variation_width VARIATION_WIDTH, -vw VARIATION_WIDTH
                        from vc sum out this many inds
                        
<add other mecana.py subcommands>
  
