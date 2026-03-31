## VSEPR Code Repository
Code repository for my research project, Developing a Computer Simulation For Estimating Lowest-Energy VSEPR Geometries Using the Thomson Problem.

### Structure
- Main directory: contains all code used to create the project, including a CLI and GUI interface, ipynb containing code to compile some data tables, and the script used to estimate the $e$ value of a lone electron pair.
- `data` directory: contains three files: `energies.csv` contains exact energies for 2 to 55 bonded electron pairs, extracted from various sources (which is directly referenced in the ipynb), `result_vsepr_cli.json` for configurations without lone pairs (Part 2), and `lone_pair_energies.json` for energy values only with lone pair combinations (Part 3).
- `paper` directory: contains the research paper for this project.
> [!WARNING]
> `result_vsepr_cli.json` is a very large file—about 15 MB in size!

### Running the Program(s)
First clone the repository:
```sh
git clone https://github.com/nijeff27/VSEPR.git
cd VSEPR
```
It is suggested to create a virtual environment via `venv` (`conda` can also work):
```sh
python -m venv .venv
# On most Unix systems (Linux / macOS)
source .venv/bin/activate
# On Windows, Powershell
source .venv\Scripts\Activate.ps1
```
Then install the required packages:
```sh
pip install -r requirements.txt
```
#### GUI (`vsepr-gui`)
Program tested on all platforms (macOS, Linux, Windows).

On Linux machines using the Wayland protocol (if not sure, run `echo "$XDG_SESSION_TYPE"`), make sure to run the program as follows (as Open3D only supports the X11 protocol as of 3/31/2026):
```sh
# On Linux, Wayland
XDG_SESSION_TYPE=x11 python vsepr-gui.py

# On macOS, Windows, X11
python vsepr-gui.py
```
#### CLI (`vsepr-cli`)
Should work on all platforms.
```sh
# Show options and help
python vsepr-cli.py --help
```
Program results (including point locations in $(\theta, \phi)$ coordinates and energy values per run) are saved in `data/result_vsepr_cli.json`. Note that the file already exists, which contains some runs; it is safe to delete this file as it will regenerate in subsequent runs.


### Credits
* `numpy` ([https://numpy.org/](https://numpy.org/))
* `open3d` ([https://www.open3d.org/](https://www.open3d.org/))
* `scipy` ([https://scipy.org/](https://scipy.org/)) (for performing linear regressions and formatting data)
* `jupyter` ([https://jupyter.org/](https://jupyter.org/)) (data analysis)
* `texlive` ([https://tug.org/texlive/](https://tug.org/texlive/))