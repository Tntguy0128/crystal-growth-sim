"""
============================================================
  Target check — does the predicted crystal match what the
  user wanted?

  The user specifies a desired spec (any subset of fields); this
  compares the measured characteristics against it and returns a
  per-criterion pass/fail report. This is the FORWARD design check
  (instant). True inverse design — auto-finding a seed for a target —
  is a later, differentiable-FNO extension.

  NSF IRES Physical AI Design Program
============================================================
"""


def check(props, target):
    """
    props  : dict from analyze.analyze()
    target : dict, any subset of:
        structure         : "single" | "poly"
        max_defects       : int
        min_crystallinity : float (0..1)
        orientation_deg   : float        (with orientation_tol, default 8)
        orientation_tol   : float
        wavelength        : float        (with wavelength_tol, default 10% rel)
        wavelength_tol    : float

    Returns (rows, all_pass) where each row is
        {criterion, desired, actual, ok}
    """
    rows = []

    def add(crit, desired, actual, ok):
        rows.append({"criterion": crit, "desired": desired,
                     "actual": actual, "ok": bool(ok)})

    if "structure" in target:
        want = target["structure"]
        g = props["n_grains"]
        is_single = g <= 1
        ok = is_single if want == "single" else (g >= 2)
        add("Structure", want,
            "single crystal" if is_single else f"polycrystal ({g})", ok)

    if "max_defects" in target:
        md = target["max_defects"]
        add("Defects", f"≤ {md}", props["n_defects"], props["n_defects"] <= md)

    if "min_crystallinity" in target:
        mc = target["min_crystallinity"]
        add("Crystallinity", f"≥ {mc:.2f}", f"{props['crystallinity']:.2f}",
            props["crystallinity"] >= mc)

    if "orientation_deg" in target:
        tol = target.get("orientation_tol", 8.0)
        want = target["orientation_deg"] % 60.0
        got = props["dominant_orientation_deg"]
        d = abs(want - got) % 60.0
        d = min(d, 60.0 - d)
        add("Orientation", f"{want:.0f}° ± {tol:.0f}°", f"{got:.1f}°", d <= tol)

    if "wavelength" in target:
        want = target["wavelength"]
        tol = target.get("wavelength_tol", 0.1 * want)
        got = props["lattice_wavelength"]
        add("Wavelength", f"{want:.2f} ± {tol:.2f}", f"{got:.2f}",
            abs(got - want) <= tol)

    all_pass = all(r["ok"] for r in rows) if rows else False
    return rows, all_pass


def report_lines(rows, all_pass):
    """Console/markdown-friendly rendering of a check() result."""
    out = []
    for r in rows:
        mark = "✓" if r["ok"] else "✗"
        out.append(f"  {mark}  {r['criterion']:<14} want {r['desired']:<16} "
                   f"got {r['actual']}")
    out.append(f"\n  {'✓ MATCHES TARGET' if all_pass else '✗ does not match target'}")
    return out


if __name__ == "__main__":
    # quick smoke test with a fabricated props dict
    props = {"n_grains": 1, "n_defects": 3, "crystallinity": 0.92,
             "dominant_orientation_deg": 14.0, "lattice_wavelength": 6.3}
    target = {"structure": "single", "max_defects": 5, "min_crystallinity": 0.8,
              "orientation_deg": 15.0, "wavelength": 6.28}
    rows, ok = check(props, target)
    print("\n".join(report_lines(rows, ok)))
    assert ok, "fabricated single crystal should match this target"
    # now demand zero defects -> should fail
    rows2, ok2 = check(props, {"max_defects": 0})
    assert not ok2
    print("\ncheck() smoke test passed.")
