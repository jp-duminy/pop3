"""

Scripts for rudimentary testing of cloudy scripts.

"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

cloudy_dir = Path.home() / "mphys" / "cloudy_files"
plots_dir = Path.home() / "mphys" / "plots"
mnras_style = Path.home() / "mnras.mplstyle"

plt.style.use(mnras_style)

datasets = {
    "out": cloudy_dir / "pop3_test.out",
    "lines": cloudy_dir / "pop3_test.lin",
    "continuum": cloudy_dir / "pop3_test.con",
}

def read_line_list(lin_path: Path) -> dict[str, np.ndarray]:
    """
    Hacked together .lin file parser.
    """
    with open(lin_path) as f:  # split by newline and tab separation
        header_fields = f.readline().rstrip("\n").split("\t")

    line_labels = [label.strip() for label in header_fields[1:] if label.strip()]

    intensities = np.loadtxt(lin_path, comments="#", delimiter="\t",
                             usecols=range(1, len(line_labels) + 1), ndmin=2)

    lines = {label: intensities[:, col_idx] for col_idx, label in enumerate(line_labels)}

    return lines

wavelength, total = np.loadtxt(datasets["continuum"], usecols=(0, 6),
                               unpack=True, comments="#", delimiter="\t")

fig, ax = plt.subplots()
in_window = (wavelength > 3500.0) & (wavelength < 7000.0)
ax.semilogy(wavelength[in_window], total[in_window], linewidth=0.6)

ax.set_xlim(3500.0, 7000.0)    

display_names = {
    "He 2 4685.68A": r"He\,\textsc{ii}",
    "Blnd 5875.55A": r"He\,\textsc{i}",
    "H  1 4861.32A": r"H$\beta$",
    "o  3 5006.84A": r"[O\,\textsc{iii}]",
}

label_transform = ax.get_xaxis_transform()   # x: data, y: axes fraction
line_intensities = read_line_list(datasets["lines"])
for line_label in line_intensities:
    line_wavelength = float(line_label.split()[-1][:-1])
    ax.plot([line_wavelength, line_wavelength], [0.80, 0.83],
            transform=label_transform, color="black", linewidth=0.6)
    ax.text(line_wavelength, 0.85, display_names[line_label],
            transform=label_transform, rotation=90, ha="center", va="bottom")

ax.set_ylabel(r"$\nu F_{\nu} [\mathrm{erg \, s^{-1}}]$")
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_title(r"Cloudy on Pop3 Test (POP3-379 at $z = 12.0$)")

fig.savefig("pop3_test.png", dpi=400)
