import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from qe_layers import parse_pw_last_structure, layers_from_z, plot_spacings

st.set_page_config(page_title="QE Interlayer Spacing", layout="centered")

st.title("QE Interlayer Spacing")
st.caption("Upload Quantum ESPRESSO pw.x output → layer means, spacings, CSV + PNG")

upl = st.file_uploader("Upload pw.x output (.out)", type=["out", "txt", "log"])
tol = st.slider("Layer clustering tolerance (Å)", 0.05, 0.60, 0.30, 0.01)
bulk_ref = st.checkbox("Show bulk Cu(111) reference ≈ 2.04 Å", value=True)

if upl:
    text = upl.read().decode("utf-8", errors="ignore")

    import tempfile, os
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".out") as f:
        f.write(text)
        f.flush()
        tmp_path = f.name

    try:
        cart, species, cell = parse_pw_last_structure(tmp_path)
        z = cart[:, 2]
        df = layers_from_z(z, tol=tol)

        # % deviation from bulk (111) optional:
        ref = 2.04
        s = df["Spacing to next (Å)"].copy()
        df["% dev vs 2.04 Å"] = s.apply(lambda v: None if pd.isna(v) else 100.0*(v-ref)/ref)

        st.subheader("Layers (bottom→top)")
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv,
                           file_name="layers_z_and_spacing.csv",
                           mime="text/csv")

        # Plot
        fig = plot_spacings(df, png_path=None)
        ax = fig.axes[0]
        if bulk_ref and df["Spacing to next (Å)"].notna().any():
            ax.axhline(ref, linestyle="--", linewidth=1)
            ax.text(0.5, ref, " bulk (111) ≈ 2.04 Å", va="bottom")
        st.pyplot(fig)

        # PNG download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        st.download_button("Download plot (PNG)", data=buf.getvalue(),
                           file_name="interlayer_distances.png",
                           mime="image/png")
    finally:
        try: os.remove(tmp_path)
        except: pass

st.markdown("---")
st.caption("Tip: If top/bottom spacings look off, increase slab thickness or relax more.")
