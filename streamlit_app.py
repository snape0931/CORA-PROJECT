"""Cora GCN Explorer: Streamlit front-end for the FastAPI service in main.py.

1. Start the API:   uvicorn main:app --reload
2. Start this UI:   streamlit run streamlit_app.py --theme.base dark
"""
import json
import math
import re
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

FEATURE_DIM = 1433
CORA_NODES = 2708
MAX_CUSTOM_NODES = 20
COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb923c", "#2dd4bf"]
DEFAULT_CLASSES = {
    0: "Case_Based", 1: "Genetic_Algorithms", 2: "Neural_Networks", 3: "Probabilistic_Methods",
    4: "Reinforcement_Learning", 5: "Rule_Learning", 6: "Theory",
}
SPARSE, DENSE = "Sparse indices", "Dense JSON"

st.set_page_config(page_title="Cora GCN Explorer", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
      div[data-testid="stMetric"] {
        background: #141c2b; border: 1px solid #25314a; border-radius: 14px; padding: 14px 18px;
      }
      .block-container { padding-top: 2rem; max-width: 1200px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers
def pretty(name):
    return str(name).replace("_", " ")


def probs_of(p):
    """Probabilities for one prediction. Works with both the new key and the old misspelled one."""
    return p.get("probabilities") or p.get("probabilites") or []


def api_get(base, path):
    r = requests.get(base.rstrip("/") + path, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(base, path, payload):
    r = requests.post(base.rstrip("/") + path, json=payload, timeout=180)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(str(detail))
    return r.json()


def render_chart(chart):
    try:
        st.altair_chart(chart, use_container_width=True)
    except TypeError:  # newer Streamlit versions dropped the argument
        st.altair_chart(chart)


def render_df(df, **kwargs):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, hide_index=True, **kwargs)


def parse_indices(text):
    out = []
    for part in re.split(r"[,\s;]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f'"{part}" is not a valid index or range.')
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if b < a:
            raise ValueError(f"The range {part} runs backwards.")
        out.extend(range(a, b + 1))
        if len(out) > 200:
            raise ValueError("Please request at most 200 papers at a time.")
    if not out:
        raise ValueError("Enter at least one paper index.")
    bad = [i for i in out if i >= CORA_NODES]
    if bad:
        raise ValueError(f"Index {bad[0]} is out of range (valid: 0-{CORA_NODES - 1}).")
    return list(dict.fromkeys(out))


def parse_nodes(text, mode):
    if not text.strip():
        raise ValueError("Add node features first, or click 'Generate random graph'.")
    if mode == SPARSE:
        vectors = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for k, line in enumerate(lines, start=1):
            vec = [0.0] * FEATURE_DIM
            for tok in re.split(r"[,\s]+", line):
                if not tok:
                    continue
                if not tok.isdigit():
                    raise ValueError(f'Line {k}: "{tok}" is not a feature index.')
                i = int(tok)
                if i >= FEATURE_DIM:
                    raise ValueError(f"Line {k}: feature index {i} is out of range (0-{FEATURE_DIM - 1}).")
                vec[i] = 1.0
            vectors.append(vec)
        return vectors
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The dense features are not valid JSON: {exc}")
    if not isinstance(parsed, list) or not parsed or not all(isinstance(v, list) for v in parsed):
        raise ValueError("Dense features must be a list of lists, one inner list per paper.")
    for k, v in enumerate(parsed):
        if len(v) != FEATURE_DIM:
            raise ValueError(f"Paper {k} has {len(v)} features; expected {FEATURE_DIM}.")
    return parsed


def parse_edges(text, n, undirected):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    seen, src, dst = set(), [], []

    def add(a, b):
        if (a, b) not in seen:
            seen.add((a, b))
            src.append(a)
            dst.append(b)

    for k, line in enumerate(lines, start=1):
        nums = re.findall(r"\d+", line)
        if len(nums) != 2:
            raise ValueError(f"Edge line {k} must contain exactly two node numbers.")
        a, b = int(nums[0]), int(nums[1])
        if a >= n or b >= n:
            raise ValueError(f"Edge line {k}: node {max(a, b)} does not exist (the graph has {n} nodes, 0-{n - 1}).")
        add(a, b)
        if undirected:
            add(b, a)
    return [src, dst]


def format_nodes(active_lists, mode):
    if mode == SPARSE:
        return "\n".join(", ".join(map(str, a)) for a in active_lists)
    rows = []
    for a in active_lists:
        v = [0] * FEATURE_DIM
        for i in a:
            v[i] = 1
        rows.append(json.dumps(v))
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def generate_graph():
    """Callback: fill the node/edge text areas with a random graph."""
    ss = st.session_state
    n = int(ss.gen_n)
    density = float(ss.gen_density) / 100
    rng = np.random.default_rng()

    active = []
    for _ in range(n):
        k = max(1, int(round(FEATURE_DIM * density * rng.uniform(0.7, 1.3))))
        active.append(sorted(rng.choice(FEATURE_DIM, size=k, replace=False).tolist()))
    ss.nodes_text = format_nodes(active, ss.feat_mode)

    edges = []
    if ss.gen_topo == "Ring" and n > 1:
        edges = [f"{i} {(i + 1) % n}" for i in range(n)]
    elif ss.gen_topo == "Random neighbours" and n > 1:
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for j in rng.choice(others, size=min(2, len(others)), replace=False).tolist():
                edges.append(f"{i} {j}")
    ss.edges_text = "\n".join(edges)
    ss.undirected = True


def randomize_cora():
    picks = sorted(np.random.choice(CORA_NODES, size=6, replace=False).tolist())
    st.session_state.cora_idx = ", ".join(map(str, picks))
    st.session_state.cora_autorun = True


# ----------------------------------------------------------------- charts
def class_color_scale(class_names):
    ids = sorted(class_names)
    return alt.Scale(
        domain=[pretty(class_names[i]) for i in ids],
        range=[COLORS[i % len(COLORS)] for i in ids],
    )


def prob_chart(p, class_names):
    probs = probs_of(p)
    ids = list(range(len(probs)))
    df = pd.DataFrame(
        {
            "Topic": [pretty(class_names.get(i, f"Class {i}")) for i in ids],
            "Probability": probs,
            "Logit": p["logits"],
        }
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Topic:N", sort="-x", title=None),
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Topic:N", scale=class_color_scale(class_names), legend=None),
            tooltip=["Topic", alt.Tooltip("Probability:Q", format=".2%"), alt.Tooltip("Logit:Q", format=".3f")],
        )
        .properties(height=230)
    )


def distribution_chart(preds, class_names):
    counts = Counter(p["predicted_class_id"] for p in preds)
    df = pd.DataFrame(
        {
            "Topic": [pretty(class_names[i]) for i in sorted(class_names)],
            "Papers": [counts.get(i, 0) for i in sorted(class_names)],
        }
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Topic:N", sort=None, title=None),
            x=alt.X("Papers:Q", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Topic:N", scale=class_color_scale(class_names), legend=None),
            tooltip=["Topic", "Papers"],
        )
        .properties(height=230)
    )


def graph_chart(preds, edges, class_names):
    n = len(preds)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False) - math.pi / 2
    x, y = np.cos(ang), np.sin(ang)
    nodes = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "Paper": list(range(n)),
            "Topic": [pretty(class_names[p["predicted_class_id"]]) for p in preds],
        }
    )
    dom = alt.Scale(domain=[-1.35, 1.35])
    ax = alt.Axis(labels=False, ticks=False, domain=False, grid=False, title=None)
    layers = []
    if edges:
        pairs = [(s, d) for s, d in zip(edges[0], edges[1]) if s != d]
        if pairs:
            seg = pd.DataFrame(
                {
                    "x": [x[s] for s, _ in pairs], "y": [y[s] for s, _ in pairs],
                    "x2": [x[d] for _, d in pairs], "y2": [y[d] for _, d in pairs],
                }
            )
            layers.append(
                alt.Chart(seg)
                .mark_rule(color="#8e9bb5", opacity=0.35)
                .encode(x=alt.X("x:Q", scale=dom, axis=ax), y=alt.Y("y:Q", scale=dom, axis=ax), x2="x2:Q", y2="y2:Q")
            )
    points = (
        alt.Chart(nodes)
        .mark_circle(size=520, stroke="#0a0e16", strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=dom, axis=ax),
            y=alt.Y("y:Q", scale=dom, axis=ax),
            color=alt.Color("Topic:N", scale=class_color_scale(class_names), legend=alt.Legend(orient="bottom")),
            tooltip=["Paper", "Topic"],
        )
    )
    labels = (
        alt.Chart(nodes)
        .mark_text(color="#0a0e16", fontWeight="bold", fontSize=12)
        .encode(x=alt.X("x:Q", scale=dom), y=alt.Y("y:Q", scale=dom), text="Paper:N")
    )
    layers += [points, labels]
    return alt.layer(*layers).properties(height=340).configure_view(strokeWidth=0)


def feature_map_image(vec, cols=48, cell=6):
    arr = np.zeros(cols * math.ceil(FEATURE_DIM / cols))
    arr[:FEATURE_DIM] = np.abs(np.asarray(vec, dtype=float))
    grid = arr.reshape(-1, cols)
    grid = np.clip(grid / (grid.max() or 1), 0, 1)
    base, accent = np.array([26, 36, 56]), np.array([108, 140, 255])
    img = base + (accent - base) * grid[..., None]
    return img.repeat(cell, axis=0).repeat(cell, axis=1).astype(np.uint8)


# ----------------------------------------------------------------- results
def show_results(data, class_names, show_logits, vectors=None, edges=None):
    preds = data["predictions"]
    confs = [max(probs_of(p)) for p in preds]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes in graph", f"{data['num_nodes']:,}")
    m2.metric("Edges in graph", f"{data['num_edges']:,}")
    m3.metric("Papers predicted", len(preds))
    m4.metric("Mean confidence", f"{np.mean(confs):.1%}")

    summary = pd.DataFrame(
        {
            "Paper": [p["node_index"] for p in preds],
            "Predicted topic": [pretty(p["predicted_class_name"]) for p in preds],
            "Confidence (%)": [c * 100 for c in confs],
        }
    )
    left, right = st.columns(2)
    with left:
        st.markdown("##### Predictions")
        render_df(
            summary,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )
    with right:
        st.markdown("##### Predicted topics")
        render_chart(distribution_chart(preds, class_names))

    if vectors is not None and len(preds) > 1:
        st.markdown("##### Graph view")
        render_chart(graph_chart(preds, edges, class_names))

    st.markdown("##### Per-paper details")
    for p in preds:
        conf = max(probs_of(p))
        title = f"Paper #{p['node_index']}  →  {pretty(p['predicted_class_name'])}  ·  {conf:.1%}"
        with st.expander(title, expanded=len(preds) <= 2):
            if vectors is not None:
                chart_col, feat_col = st.columns([3, 2])
            else:
                chart_col, feat_col = st.container(), None
            with chart_col:
                render_chart(prob_chart(p, class_names))
                if show_logits:
                    render_df(
                        pd.DataFrame(
                            {
                                "Topic": [pretty(class_names.get(i, i)) for i in range(len(p["logits"]))],
                                "Logit": p["logits"],
                                "Probability": probs_of(p),
                            }
                        )
                    )
            if feat_col is not None:
                with feat_col:
                    vec = vectors[p["node_index"]]
                    active = [i for i, v in enumerate(vec) if v]
                    st.image(feature_map_image(vec), width=288)
                    st.caption(f"{len(active)} of {FEATURE_DIM} features active ({len(active) / FEATURE_DIM:.1%})")
                    if active:
                        shown = ", ".join(f"#{i}" for i in active[:14])
                        extra = f" +{len(active) - 14} more" if len(active) > 14 else ""
                        st.code(shown + extra, language=None)

    with st.expander("Raw JSON response"):
        st.json(data)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("◈ Cora GCN")
    st.caption("Classify research papers by topic with a graph neural network.")
    api_url = st.text_input("FastAPI URL", "http://127.0.0.1:8000")
    show_logits = st.toggle("Show raw logits", value=False)

    info = None
    try:
        health = api_get(api_url, "/health")
        info = api_get(api_url, "/info")
        st.success(f"API online · {', '.join(health.get('providers', []))}")
    except Exception as exc:
        st.error(f"Cannot reach the API: {exc}")

if info is None:
    st.warning("Start the API first (`uvicorn main:app --reload`), then check the URL in the sidebar.")
    st.stop()

class_names = {int(k): v for k, v in info.get("class_mapping", DEFAULT_CLASSES).items()}

# ----------------------------------------------------------------- state defaults
ss = st.session_state
ss.setdefault("cora_idx", "0, 1, 2, 3, 4, 5")
ss.setdefault("gen_n", 6)
ss.setdefault("gen_density", 1.3)
ss.setdefault("gen_topo", "Random neighbours")
ss.setdefault("feat_mode", SPARSE)
ss.setdefault("undirected", True)
if "nodes_text" not in ss:
    generate_graph()

st.title("Cora GCN Explorer")
tab_cora, tab_custom, tab_info = st.tabs(["Cora papers", "Custom graph", "Model info"])

# ----------------------------------------------------------------- tab 1
with tab_cora:
    st.subheader("Predict real Cora papers")
    st.caption(f"Enter paper indices from 0 to {CORA_NODES - 1}. Commas and ranges work, e.g. 0, 5, 10-15.")
    c_in, c_go, c_rand = st.columns([6, 1, 1])
    c_in.text_input("Paper indices", key="cora_idx", label_visibility="collapsed")
    go = c_go.button("Predict", type="primary", key="cora_go")
    c_rand.button("Random 6", on_click=randomize_cora, key="cora_rand")

    if go or ss.pop("cora_autorun", False):
        try:
            idx = parse_indices(ss.cora_idx)
            with st.spinner("Running the GCN… the first request loads the Cora dataset."):
                ss.cora_result = api_post(api_url, "/predict/cora_node", {"node_indices": idx})
        except Exception as exc:
            ss.pop("cora_result", None)
            st.error(str(exc))

    if "cora_result" in ss:
        show_results(ss.cora_result, class_names, show_logits)

# ----------------------------------------------------------------- tab 2
with tab_custom:
    st.subheader("Build a custom graph")
    st.caption("Every paper has 1433 binary word features. Generate a random graph, then edit it by hand if you like.")

    g1, g2, g3, g4 = st.columns([1, 2, 2, 2])
    g1.number_input("Papers", min_value=1, max_value=MAX_CUSTOM_NODES, key="gen_n")
    g2.slider("Feature density (%)", min_value=0.3, max_value=5.0, step=0.1, key="gen_density")
    g3.selectbox("Citations", ["Random neighbours", "Ring", "None (self-loops only)"], key="gen_topo")
    g4.radio("Feature format", [SPARSE, DENSE], key="feat_mode", horizontal=True)
    st.button("Generate random graph", on_click=generate_graph, key="gen_btn")

    t1, t2 = st.columns(2)
    with t1:
        hint = (
            "One paper per line: the indices (0-1432) of the words that appear."
            if ss.feat_mode == SPARSE
            else "A JSON list with one vector of exactly 1433 numbers per paper."
        )
        st.text_area("Node features", key="nodes_text", height=240, help=hint)
        st.caption(hint + " After switching format, click Generate again.")
    with t2:
        st.text_area("Citations (edges)", key="edges_text", height=240, help="One edge per line: source target")
        st.checkbox("Treat edges as undirected", key="undirected")
        st.caption("Leave empty to use self-loops only.")

    if st.button("Predict graph", type="primary", key="custom_go"):
        try:
            vectors = parse_nodes(ss.nodes_text, ss.feat_mode)
            edges = parse_edges(ss.edges_text, len(vectors), ss.undirected)
            with st.spinner("Running the GCN…"):
                data = api_post(api_url, "/predict", {"node_features": vectors, "edge_indices": edges})
            ss.custom_result = (data, vectors, edges)
        except Exception as exc:
            ss.pop("custom_result", None)
            st.error(str(exc))

    if "custom_result" in ss:
        data, vectors, edges = ss.custom_result
        show_results(data, class_names, show_logits, vectors=vectors, edges=edges)

# ----------------------------------------------------------------- tab 3
with tab_info:
    st.subheader(info.get("Model Name", "Model"))
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Feature dimension", info.get("feature_dimension"))
    i2.metric("Classes", info.get("num_classes"))
    i3.metric("ONNX inputs", len(info.get("inputs", [])))
    i4.metric("ONNX outputs", len(info.get("outputs", [])))

    st.markdown("##### Topic classes")
    render_df(pd.DataFrame({"ID": list(class_names), "Topic": [pretty(v) for v in class_names.values()]}))

    for key, label in (("inputs", "Inputs"), ("outputs", "Outputs")):
        st.markdown(f"##### {label}")
        rows = [{"Name": t["name"], "Shape": str(t["shape"]), "Type": t["type"]} for t in info.get(key, [])]
        render_df(pd.DataFrame(rows))
