"""
Passive GDS layout of the 4x4 Clements-mesh programmable photonic processor.

Every MZI's 2x2 splitter and combiner is the FEM-designed directional coupler
(200 nm gap, ~18.9 um length for 50:50 at 1550 nm; see fdtd/coupler_modes.py).
The 6 MZIs form the rectangular Clements mesh for N=4:

    column 0 : MZI on modes (0,1) and (2,3)
    column 1 : MZI on modes (1,2)
    column 2 : MZI on modes (0,1) and (2,3)
    column 3 : MZI on modes (1,2)
    -----------------------------------------  6 MZIs = N(N-1)/2

The 4 optical modes travel on fixed horizontal y-lines. In each column, a mode
either enters an MZI (mapped to that MZI's upper/lower port) or is carried
straight across by a pass-through waveguide. Grating couplers terminate all 4
inputs and 4 outputs.

PASSIVE layout (no heaters/metal yet -- that is the next layer). Goal: a valid,
DRC-checkable GDS with correct connectivity built from the FEM coupler.

Run:  python layout/mesh_layout.py
Out:  results/clements_mesh_4x4.gds , results/clements_mesh_4x4.png
"""

import os
import gdsfactory as gf
from gdsfactory.gpdk import get_generic_pdk

get_generic_pdk().activate()

OUT = "results"
os.makedirs(OUT, exist_ok=True)

DC_GAP = 0.20       # um  (FEM design)
DC_LEN = 18.9      # um  (FEM 50:50 length)
PITCH = 60.0       # vertical mode spacing [um] (clears GC taper width)
COL_X = 200.0      # horizontal column spacing [um]
MZI_LEN = 127.8    # measured MZI footprint length [um]

# Clements N=4: (column, upper_mode) for each MZI
MESH = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1)]
N_MODES = 4
N_COLS = 4


@gf.cell
def fem_coupler() -> gf.Component:
    return gf.components.coupler(gap=DC_GAP, length=DC_LEN)


@gf.cell
def mzi_2x2(delta_length: float = 20.0) -> gf.Component:
    return gf.components.mzi2x2_2x2(
        splitter=fem_coupler(), combiner=fem_coupler(),
        delta_length=delta_length, cross_section="strip",
    )


def mode_y(mode: int) -> float:
    """Fixed y-line for a mode (mode 0 at top, y decreasing downward)."""
    return -mode * PITCH


@gf.cell
def clements_mesh_4x4(delta_length: float = 20.0) -> gf.Component:
    c = gf.Component()
    mzi = mzi_2x2(delta_length=delta_length)

    # ---- place MZIs so their two modes land on the correct y-lines ----
    # upper port (o2/o3, y=+2.35) -> upper_mode ; lower port (o1/o4) -> upper_mode+1
    refs = {}
    for (col, um) in MESH:
        r = c.add_ref(mzi)
        # want upper input o2 at y = mode_y(um); midline between modes um, um+1
        target_upper_y = mode_y(um)
        dy = target_upper_y - r.ports["o2"].dcenter[1]
        x_left = col * COL_X
        dx = x_left - r.ports["o2"].dcenter[0]
        r.move((dx, dy))
        refs[(col, um)] = r

    # left/right port for a given (col, mode) if an MZI there carries it
    def ports_at(col, mode):
        for (mc, um) in MESH:
            if mc != col:
                continue
            if um == mode:            # upper line of this MZI
                return refs[(col, um)].ports["o2"], refs[(col, um)].ports["o3"]
            if um + 1 == mode:        # lower line of this MZI
                return refs[(col, um)].ports["o1"], refs[(col, um)].ports["o4"]
        return None, None

    xs = "strip"
    x_entry = -70.0
    x_exit = (N_COLS - 1) * COL_X + MZI_LEN + 70.0

    # ---- route each mode across the columns on its fixed y-line ----
    for mode in range(N_MODES):
        y = mode_y(mode)
        c.add_port(name=f"in_{mode}", center=(x_entry, y), width=0.5,
                   orientation=180, layer=gf.get_cross_section(xs).layer)
        prev_right = c.ports[f"in_{mode}"]

        for col in range(N_COLS):
            lp, rp = ports_at(col, mode)
            if lp is None:
                continue  # handled as one long straight after the loop
            gf.routing.route_single(c, prev_right, lp, cross_section=xs)
            prev_right = rp

        c.add_port(name=f"out_{mode}", center=(x_exit, y), width=0.5,
                   orientation=0, layer=gf.get_cross_section(xs).layer)
        gf.routing.route_single(c, prev_right, c.ports[f"out_{mode}"], cross_section=xs)

    # ---- grating couplers on all 8 I/O ----
    gc = gf.components.grating_coupler_elliptical_te()
    for mode in range(N_MODES):
        c.add_ref(gc).connect("o1", c.ports[f"in_{mode}"])
        c.add_ref(gc).connect("o1", c.ports[f"out_{mode}"])

    return c


if __name__ == "__main__":
    print("Building 4x4 Clements mesh (passive)...")
    mesh = clements_mesh_4x4()
    gds_path = os.path.join(OUT, "clements_mesh_4x4.gds")
    mesh.write_gds(gds_path)
    print(f"  wrote {gds_path}")
    print(f"  MZIs placed : {len(MESH)} (expected 6)")
    bb = mesh.dbbox()
    print(f"  footprint   : {bb.width():.0f} x {bb.height():.0f} um")
