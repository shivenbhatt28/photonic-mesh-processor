"""
Geometric DRC on the mesh GDS using KLayout's DRC engine.

Checks the waveguide layer (1/0) for minimum width and minimum spacing against
representative generic-SOI strip rules. Reports violation counts and writes an
error-marker GDS you can open next to the layout in KLayout to see exactly
where each violation is.

Note on interpreting results: directional couplers intentionally bring
waveguides to a 200 nm gap, so a spacing check at/above 200 nm will flag those
by design. The meaningful check is *below* the design gap (e.g. 150 nm), which
catches genuine near-collisions from routing rather than the intended coupler
gaps.

Run:  python layout/drc_check.py
"""

import os
import sys
import klayout.db as kdb

GDS = os.path.join("results", "clements_mesh_4x4.gds")
WG_LAYER = (1, 0)
MIN_WIDTH_UM = 0.180
MIN_SPACE_UM = 0.150      # below the intentional 200 nm coupler gap


def run(gds_path: str = GDS) -> int:
    if not os.path.exists(gds_path):
        print(f"GDS not found: {gds_path} (run layout/mesh_layout.py first)")
        return 2
    ly = kdb.Layout()
    ly.read(gds_path)
    dbu = ly.dbu
    top = ly.top_cell()
    wg = ly.layer(*WG_LAYER)

    reg = kdb.Region(top.begin_shapes_rec(wg))
    reg.merge()

    w_viol = reg.width_check(int(MIN_WIDTH_UM / dbu))
    s_viol = reg.space_check(int(MIN_SPACE_UM / dbu))

    print(f"GDS            : {gds_path}")
    print(f"top cell       : {top.name}")
    print(f"WG polygons    : {reg.count()}")
    print(f"WG area        : {reg.area() * dbu * dbu:.1f} um^2")
    print(f"width  < {MIN_WIDTH_UM*1000:.0f} nm : {w_viol.count()} violations")
    print(f"space  < {MIN_SPACE_UM*1000:.0f} nm : {s_viol.count()} violations")

    # write error markers for inspection in KLayout
    out = kdb.Layout()
    out.dbu = dbu
    tc = out.create_cell("DRC_ERRORS")
    tc.shapes(out.layer(1000, 0)).insert(w_viol)
    tc.shapes(out.layer(1001, 0)).insert(s_viol)
    marker_path = os.path.join("results", "drc_errors.gds")
    out.write(marker_path)
    print(f"error markers  : {marker_path} "
          f"(layers 1000=width, 1001=space)")

    return 0 if (w_viol.count() == 0 and s_viol.count() == 0) else 1


if __name__ == "__main__":
    sys.exit(run())
