import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pulp
from itertools import combinations_with_replacement
from io import BytesIO

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username == "vishal" and password == "vishal":
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password ❌")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Check login state
if not st.session_state["logged_in"]:
    login()
    st.stop()
else:
    st.sidebar.success(f"✅ Logged in as {st.session_state['username']}")
    st.markdown(f"### Welcome {st.session_state['username'].capitalize()} 👋")

st.set_page_config(layout="wide")
st.title("Tube Cutting Optimization — Minimize Waste (pattern-based)")

# ---------------------- Template Excel ----------------------
def generate_template_bytes():
    """
    Single-sheet template. Columns:
      Tube Length (mm), Kerf (mm), Diameter (mm), Thickness (mm),
      Cut Length (mm), Quantity
    """
    df = pd.DataFrame({
        "Tube Length (mm)": [6000, 6000, 6000],
        "Kerf (mm)": [8, 8, 8],
        "Diameter (mm)": [50, 50, 50],
        "Thickness (mm)": [2, 2, 2],
        "Cut Length (mm)": [1512, 1297, 732],
        "Quantity": [18, 8, 8]
    })
    output = BytesIO()
    # Use openpyxl to avoid xlsxwriter dependency issues
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cutting Plan")
    return output.getvalue()

# ---------------------- Sidebar ----------------------
st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("📂 Upload Cutting Plan Excel (single sheet)", type=["xlsx"])
st.sidebar.download_button(
    label="📥 Download Template Excel",
    data=generate_template_bytes(),
    file_name="cutting_plan_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.sidebar.markdown("---")
st.sidebar.write("Template columns (single sheet 'Cutting Plan'):")
st.sidebar.code("Tube Length (mm), Kerf (mm), Diameter (mm), Thickness (mm), Cut Length (mm), Quantity")
st.sidebar.markdown("---")
util_thresh = st.sidebar.slider("Minimum pattern utilization (%) to keep pattern", 20, 80, 50)
max_patterns_per_group = st.sidebar.number_input("Max patterns per spec (to limit runtime)", min_value=100, max_value=20000, value=3000, step=100)

# ---------------------- Load Input ----------------------
if uploaded_file is not None:
    try:
        cutting_plan = pd.read_excel(uploaded_file, sheet_name="Cutting Plan")
        st.success("Excel file loaded successfully ✅")
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()
else:
    st.header("Daily Cutting Plan (Manual Input)")
    default_data = {
        "Tube Length (mm)": [6000, 6000, 6500],
        "Kerf (mm)": [8, 8, 10],
        "Diameter (mm)": [50, 50, 75],
        "Thickness (mm)": [2, 2, 3],
        "Cut Length (mm)": [1512, 1297, 732],
        "Quantity": [18, 8, 8]
    }
    cutting_plan = pd.DataFrame(default_data)
    cutting_plan = st.data_editor(cutting_plan, num_rows="dynamic")

# ---------------------- Utility functions ----------------------
def clean_plan(df):
    df = df.copy().dropna(how="all")
    numeric_cols = ["Tube Length (mm)", "Kerf (mm)", "Diameter (mm)", "Thickness (mm)", "Cut Length (mm)", "Quantity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # cast to int where appropriate
    df["Tube Length (mm)"] = df["Tube Length (mm)"].astype(int)
    df["Kerf (mm)"] = df["Kerf (mm)"].astype(int)
    df["Diameter (mm)"] = df["Diameter (mm)"].astype(int)
    df["Thickness (mm)"] = df["Thickness (mm)"].astype(int)
    df["Cut Length (mm)"] = df["Cut Length (mm)"].astype(int)
    df["Quantity"] = df["Quantity"].astype(int)
    df = df[(df["Cut Length (mm)"] > 0) & (df["Quantity"] > 0)]
    return df

def generate_patterns(cut_lengths, demands, tube_len, kerf, max_patterns=3000, util_threshold=0.5):
    """
    Generate feasible patterns for given cut_lengths (list) and per-length demands (list).
    - cuts_lengths: distinct lengths list (sorted descending)
    - demands: corresponding required quantities
    - Limits:
        * For each length L, max_repeat = min(demand[L], floor((tube_len + kerf)/(L+kerf)))
    - Generate combinations_with_replacement up to max_pieces and filter:
        * total_used = sum(combo) + (len(combo)-1)*kerf <= tube_len
        * utilization = total_used / tube_len >= util_threshold
    Returns list of pattern dicts: {counts, combo, used, utilization}
    """
    # ensure lists
    cut_lengths = list(cut_lengths)
    demands = list(demands)
    # sort lengths descending for better pruning (keep mapping)
    order = sorted(range(len(cut_lengths)), key=lambda i: -cut_lengths[i])
    cut_lengths = [cut_lengths[i] for i in order]
    demands = [demands[i] for i in order]

    # compute per-length max repeats allowed
    max_repeats = []
    for L, d in zip(cut_lengths, demands):
        if L + kerf <= 0:
            max_repeats.append(0)
        else:
            rep_limit = int((tube_len + kerf) // (L + kerf))
            max_repeats.append(min(d, rep_limit))

    # if any max_repeats zero but demand >0 and L>tube_len -> impossible handled elsewhere
    if len(cut_lengths) == 0:
        return [], cut_lengths

    # compute max pieces in a tube (upper bound)
    minL = min([L for L in cut_lengths if L > 0])
    max_pieces = int((tube_len + kerf) // (minL + kerf))
    max_pieces = max(1, max_pieces)

    patterns = []
    seen_counts = set()

    # generate combos by number of pieces
    for n in range(1, max_pieces + 1):
        # generate combinations_with_replacement of lengths (values)
        for combo in combinations_with_replacement(cut_lengths, n):
            # quick per-length repeat check against max_repeats
            valid = True
            for idx, L in enumerate(cut_lengths):
                if combo.count(L) > max_repeats[idx]:
                    valid = False
                    break
            if not valid:
                continue
            total_used = sum(combo) + (len(combo) - 1) * kerf
            if total_used > tube_len:
                continue
            utilization = total_used / tube_len
            if utilization < util_threshold:
                continue
            # counts as tuple aligned to cut_lengths order
            counts = tuple(combo.count(L) for L in cut_lengths)
            if counts not in seen_counts:
                seen_counts.add(counts)
                patterns.append({
                    "counts": counts,
                    "combo": combo,
                    "used": total_used,
                    "utilization": utilization
                })
                if len(patterns) >= max_patterns:
                    return patterns, cut_lengths
    # If no pattern passes utilization filter, relax threshold and allow single-piece patterns
    if not patterns:
        for idx, L in enumerate(cut_lengths):
            if L <= tube_len and demands[idx] > 0:
                counts = tuple(1 if i == idx else 0 for i in range(len(cut_lengths)))
                total_used = L
                patterns.append({
                    "counts": counts,
                    "combo": (L,),
                    "used": total_used,
                    "utilization": total_used / tube_len
                })
                if len(patterns) >= max_patterns:
                    break
    return patterns, cut_lengths

def solve_cutting_stock_group(group_df, tube_len, kerf, util_threshold, max_patterns):
    """
    group_df contains rows for one tube spec (Cut Length (mm), Quantity).
    Returns:
      df_out (tubes), tubes_used, total_waste, utilization
    """
    plan = group_df.copy()
    # aggregate by cut length
    plan = plan.groupby("Cut Length (mm)", as_index=False)["Quantity"].sum().sort_values(by="Cut Length (mm)", ascending=False)
    cut_lengths = plan["Cut Length (mm)"].tolist()
    demands = plan["Quantity"].tolist()

    # sanity: remove lengths > tube_len
    for L, q in zip(cut_lengths, demands):
        if L > tube_len:
            st.warning(f"Cut length {L} mm > tube length {tube_len} mm — will be ignored.")
    feasible = [(L, q) for L, q in zip(cut_lengths, demands) if L <= tube_len and q > 0]
    if not feasible:
        return pd.DataFrame(), 0, 0, 0

    cut_lengths, demands = zip(*feasible)
    cut_lengths, demands = list(cut_lengths), list(demands)

    patterns, ordered_lengths = generate_patterns(cut_lengths, demands, tube_len, kerf,
                                                 max_patterns=max_patterns, util_threshold=util_threshold/100.0)
    if not patterns:
        return pd.DataFrame(), 0, 0, 0

    # Build integer LP: minimize total waste = sum_p y_p * (tube_len - used_p)
    prob = pulp.LpProblem("CuttingStock_MinimizeWaste", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", lowBound=0, cat="Integer") for i in range(len(patterns))]

    # objective
    waste_per_pattern = [tube_len - patt["used"] for patt in patterns]
    prob += pulp.lpSum(y[i] * waste_per_pattern[i] for i in range(len(patterns)))

    # demand constraints
    for i in range(len(ordered_lengths)):
        prob += pulp.lpSum(y[j] * patterns[j]["counts"][i] for j in range(len(patterns))) >= demands[i]

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    pattern_counts = [int(pulp.value(var) or 0) for var in y]

    tubes = []
    total_cut_pieces_length = 0
    total_waste = 0
    total_tubes_used = 0

    for p_idx, usage in enumerate(pattern_counts):
        if usage <= 0:
            continue
        patt = patterns[p_idx]
        for _ in range(usage):
            combo = patt["combo"]
            pieces_len = sum(combo)
            kerf_total = (len(combo) - 1) * kerf if len(combo) > 1 else 0
            used = pieces_len + kerf_total
            waste = tube_len - used
            tubes.append({
                "Cuts": list(combo),
                "Cut Pieces Length (mm)": pieces_len,
                "Kerf Loss (mm)": kerf_total,
                "Used (mm)": used,
                "Waste (mm)": waste
            })
            total_cut_pieces_length += pieces_len
            total_waste += waste
            total_tubes_used += 1

    if total_tubes_used == 0:
        return pd.DataFrame(), 0, 0, 0

    df_out = pd.DataFrame([{"Tube": i + 1, **t} for i, t in enumerate(tubes)])
    utilization = (total_cut_pieces_length / (total_tubes_used * tube_len)) * 100.0
    return df_out, total_tubes_used, total_waste, utilization

# ---------------------- Run Optimization ----------------------
if st.button("Optimize Cutting Plan"):
    cleaned = clean_plan(cutting_plan)
    if cleaned.empty:
        st.warning("Please provide valid cut lengths and quantities.")
    else:
        # Group by tube spec (Tube Length, Kerf, Diameter, Thickness)
        with st.spinner("Generating patterns and solving optimization per tube specification..."):
            results_per_spec = []
            summary_rows = []
            grouped = cleaned.groupby(["Tube Length (mm)", "Kerf (mm)", "Diameter (mm)", "Thickness (mm)"], sort=False)
            for spec_vals, group in grouped:
                tube_len, kerf, dia, thick = spec_vals
                spec_label = f"L{tube_len}-K{kerf}-D{dia}-T{thick}"
                # subset group to only Cut Length and Quantity
                group_subset = group[["Cut Length (mm)", "Quantity"]]
                df_out, tubes_used, total_waste, utilization = solve_cutting_stock_group(
                    group_subset, tube_len, kerf, util_thresh, max_patterns_per_group
                )
                if df_out is not None and not df_out.empty:
                    # add spec label to df_out
                    df_out.insert(0, "Spec Label", spec_label)
                    # record spec details
                    df_out["Tube Length (mm)"] = tube_len
                    df_out["Kerf (mm)"] = kerf
                    df_out["Diameter (mm)"] = dia
                    df_out["Thickness (mm)"] = thick
                    results_per_spec.append(df_out)
                    summary_rows.append({
                        "Tube Spec": spec_label,
                        "Tubes Used": tubes_used,
                        "Total Waste (mm)": total_waste,
                        "Utilization (%)": round(utilization, 2)
                    })
                else:
                    summary_rows.append({
                        "Tube Spec": spec_label,
                        "Tubes Used": 0,
                        "Total Waste (mm)": 0,
                        "Utilization (%)": 0.0
                    })

        if not results_per_spec:
            st.error("No feasible solutions found for any tube specification.")
        else:
            result_df = pd.concat(results_per_spec, ignore_index=True)
            summary_df = pd.DataFrame(summary_rows)

            st.subheader("Summary by Tube Specification")
            st.dataframe(summary_df)

            st.subheader("Optimized Cutting Plan (per tube)")
            st.dataframe(result_df[["Spec Label", "Tube", "Cuts", "Cut Pieces Length (mm)", "Kerf Loss (mm)", "Used (mm)", "Waste (mm)"]])

            # Visualization: stack per spec+tube
            viz_rows = []
            for _, row in result_df.iterrows():
                start = 0
                tube_name = f"{row['Spec Label']} - Tube {row['Tube']}"
                for cut in row["Cuts"]:
                    viz_rows.append({
                        "Tube": tube_name,
                        "Segment": f"{cut} mm",
                        "Length": cut,
                        "Type": f"Cut {cut} mm",
                        "Start": start,
                        "End": start + cut
                    })
                    start += cut
                # kerf block (optional, included in used if >0)
                if row["Kerf Loss (mm)"] > 0:
                    viz_rows.append({
                        "Tube": tube_name,
                        "Segment": f"Kerf {int(row['Kerf Loss (mm)'])} mm",
                        "Length": int(row["Kerf Loss (mm)"]),
                        "Type": "Kerf",
                        "Start": start,
                        "End": start + int(row["Kerf Loss (mm)"])
                    })
                    start += int(row["Kerf Loss (mm)"])
                if row["Waste (mm)"] > 0:
                    viz_rows.append({
                        "Tube": tube_name,
                        "Segment": "Waste",
                        "Length": int(row["Waste (mm)"]),
                        "Type": "Waste",
                        "Start": start,
                        "End": start + int(row["Waste (mm)"])
                    })

            viz_df = pd.DataFrame(viz_rows)
            if not viz_df.empty:
                # Force only Waste to be red; leave cuts/kerf to plotly palette
                fig = px.bar(
                    viz_df,
                    x="Tube",
                    y="Length",
                    color="Type",
                    text="Segment",
                    title="Tube Cutting Layout by Spec/Tube",
                    hover_data=["Start", "End"],
                    color_discrete_map={"Waste": "red"}
                )
                fig.update_layout(barmode="stack", xaxis={"categoryorder": "category ascending"}, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            # Excel export - use openpyxl engine
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                cleaned.to_excel(writer, index=False, sheet_name="Input Plan")
                result_df.to_excel(writer, index=False, sheet_name="Optimized Plan")
                summary_df.to_excel(writer, index=False, sheet_name="Summary")
            st.download_button(
                label="📥 Download Excel Report (Input + Optimized + Summary)",
                data=output.getvalue(),
                file_name="optimized_cutting_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("Optimization complete ✅")
