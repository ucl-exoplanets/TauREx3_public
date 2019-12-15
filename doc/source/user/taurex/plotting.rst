.. _userplotting:

========
Plotting
========

Along with the *--plot* argument for taurex, there
is also an extra program specifically for plotting to PDF
from TauREx 3 HDF5 outputs. It is accessed like this::

    taurex-plot

A summary of the arguments is given here:

+---------------+--------------------+-------------+-----------------------------------+   
| Argument      |  Alternate name    | Input       |  Description                      |
+---------------+--------------------+-------------+-----------------------------------+  
| -h            |  --help            |             |  show this help message and exit  |
+---------------+--------------------+-------------+-----------------------------------+
| -i            |  --input           |  INPUT_FILE |  TauREx 3 HDF5 output file        |
+---------------+--------------------+-------------+-----------------------------------+
| -o            |  --output-dir      | OUTPUT_DIR  |  Directory to store plots         |
+---------------+--------------------+-------------+-----------------------------------+
| -T            |  --title           |   TITLE     |  Title of plots (optional)        |
+---------------+--------------------+-------------+-----------------------------------+
| -p            |  --prefix          |  PREFIX     |  Output filename prefix (optional)|
+---------------+--------------------+-------------+-----------------------------------+
| -m            |  --color-map       |   CMAP      |  Matplotlib colormap (optional)   |
+---------------+--------------------+-------------+-----------------------------------+
| -R            |  --resolution      | RESOLUTION  |  Resample spectra at resolution   |
+---------------+--------------------+-------------+-----------------------------------+
| -P            |  --plot-posteriors |             |  Plot posteriors                  |
+---------------+--------------------+-------------+-----------------------------------+
| -x            |  --plot-xprofile   |             |  Plot molecular profiles          |
+---------------+--------------------+-------------+-----------------------------------+
| -t            |  --plot-tpprofile  |             |  Plot Temperature profiles        |
+---------------+--------------------+-------------+-----------------------------------+
| -d            |  --plot-tau        |             |  Plot optical depth contribution  |
+---------------+--------------------+-------------+-----------------------------------+
| -s            |  --plot-spectrum   |             |  Plot spectrum                    |
+---------------+--------------------+-------------+-----------------------------------+
| -c            |  --plot-contrib    |             |  Plot contrib                     |
+---------------+--------------------+-------------+-----------------------------------+
| -C            |  --full-contrib    |             |  Plot detailed contribs           |
+---------------+--------------------+-------------+-----------------------------------+
| -a            |  --all             |             |  Plot everythiong                 |
+---------------+--------------------+-------------+-----------------------------------+
