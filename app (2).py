import re
import math
import unicodedata
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd
import streamlit as st
from networkx.algorithms import bipartite
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Interactive PON / Plan Social Network Analysis",
    page_icon="🕸️",
    layout="wide",
)

# ============================================================
# CONFIGURATION
# ============================================================

OID_COL = "OID"
CID_COL = "CID"
ORG_NAME_COL = "standardizedOrgName"
NETWORK_NAME_COL = "networkName"
NATIVE_COL = "nativeAmerican"
ORG_IS_PON_COL = "OrgIsPON"

IDAHO_LOCAL_COL = "locatedInIdahoLans"
MONTANA_LOCAL_COL = "locatedInMontanaLans"
NEZ_PERCE_LOCAL_COL = "locatedInNZRes"
FLATHEAD_LOCAL_COL = "locatedInFHRes"
COUNTY_NAME_COL = "locationCntyName"
STATE_NAME_COL = "locationStateName"
NETWORK_SCOPE_INTEREST_COL = "networkGeoScopeOfInterest"

FUNDING_COLS = {
    "Flathead Reservation": "localFundingPost2019_FHRes",
    "Idaho County": "localFundingPost2019_IdahoCnty",
    "Lake County": "localFundingPost2019_LakeCnty",
    "Missoula County": "localFundingPost2019_MissoulaCnty",
    "Nez Perce County": "localFundingPost2019_NPCnty",
    "Nez Perce Reservation": "localFundingPost2019_NPRes",
}

GEOGRAPHIES = {
    "Missoula County": {
        "geo_type": "county",
        "landscape_col": MONTANA_LOCAL_COL,
        "county_name": "Missoula",
        "state_name": "Montana",
        "scope_labels": ["Missoula County"],
        "dark": "#D75F4C",
        "light": "#F6A482",
    },
    "Lake County": {
        "geo_type": "county",
        "landscape_col": MONTANA_LOCAL_COL,
        "county_name": "Lake",
        "state_name": "Montana",
        "scope_labels": ["Lake County"],
        "dark": "#D75F4C",
        "light": "#F6A482",
    },
    "Idaho County": {
        "geo_type": "county",
        "landscape_col": IDAHO_LOCAL_COL,
        "county_name": "Idaho",
        "state_name": "Idaho",
        "scope_labels": ["Idaho County"],
        "dark": "#3A93C3",
        "light": "#8EC4DE",
    },
    "Nez Perce County": {
        "geo_type": "county",
        "landscape_col": IDAHO_LOCAL_COL,
        "county_name": "Nez Perce",
        "state_name": "Idaho",
        "scope_labels": ["Nez Perce County"],
        "dark": "#3A93C3",
        "light": "#8EC4DE",
    },
    "Flathead Reservation": {
        "geo_type": "reservation",
        "reservation_col": FLATHEAD_LOCAL_COL,
        "scope_labels": ["Flathead Reservation", "CSKT"],
        "dark": "#D77B4C",
        "light": "#F6BC82",
    },
    "Nez Perce Reservation": {
        "geo_type": "reservation",
        "reservation_col": NEZ_PERCE_LOCAL_COL,
        "scope_labels": ["Nez Perce Reservation", "Nez Perce Tribe"],
        "dark": "#3AA9C3",
        "light": "#8ED9DE",
    },
}

LOCAL_NATIVE_COLOR = "#7B4FA3"
NONLOCAL_NATIVE_COLOR = "#B9A0D3"
LOCAL_PON_ORG_COLOR = "#D4A017"
NONLOCAL_PON_ORG_COLOR = "#F6D55C"
REGULAR_PON_COLOR = "#8A8A8A"
PLAN_COLOR = "#A8A8A8"
STANDARD_EDGE_COLOR = "#707070"
PON_TO_PON_EDGE_COLOR = "#B7791F"

REQUIRED_COLUMNS = [
    OID_COL, CID_COL, ORG_NAME_COL, NETWORK_NAME_COL,
    NATIVE_COL, ORG_IS_PON_COL,
    IDAHO_LOCAL_COL, MONTANA_LOCAL_COL,
    NEZ_PERCE_LOCAL_COL, FLATHEAD_LOCAL_COL,
    COUNTY_NAME_COL, STATE_NAME_COL,
    NETWORK_SCOPE_INTEREST_COL,
    *FUNDING_COLS.values(),
]


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def flag_to_int(value):
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) == 1 else 0
    return 1 if clean_text(value).lower() in {"1", "true", "yes", "y"} else 0


def normalize_name(value):
    text = clean_text(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_scope_labels(value):
    value = clean_text(value)
    if not value:
        return set()
    return {x.strip().casefold() for x in re.split(r"[,;]+", value) if x.strip()}


def money(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def safe_constraint(projection):
    if projection.number_of_nodes() == 0 or projection.number_of_edges() == 0:
        return {n: float("nan") for n in projection.nodes()}
    try:
        return nx.constraint(projection)
    except Exception:
        return {n: float("nan") for n in projection.nodes()}


# ============================================================
# DATA PREPARATION
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_data(file_bytes):
    import io
    df = pd.read_excel(io.BytesIO(file_bytes))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    for col in [OID_COL, CID_COL, ORG_NAME_COL, NETWORK_NAME_COL,
                COUNTY_NAME_COL, STATE_NAME_COL, NETWORK_SCOPE_INTEREST_COL]:
        df[col] = df[col].apply(clean_text)

    for col in [NATIVE_COL, ORG_IS_PON_COL, IDAHO_LOCAL_COL, MONTANA_LOCAL_COL,
                NEZ_PERCE_LOCAL_COL, FLATHEAD_LOCAL_COL]:
        df[col] = df[col].apply(flag_to_int)

    for col in FUNDING_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    oid_df = df[df[OID_COL] != ""].copy()
    cid_df = df[df[CID_COL] != ""].copy()

    org_name = (
        oid_df[oid_df[ORG_NAME_COL] != ""]
        .drop_duplicates(OID_COL)
        .set_index(OID_COL)[ORG_NAME_COL]
        .to_dict()
    )
    network_name = (
        cid_df[cid_df[NETWORK_NAME_COL] != ""]
        .drop_duplicates(CID_COL)
        .set_index(CID_COL)[NETWORK_NAME_COL]
        .to_dict()
    )

    native_lookup = oid_df.groupby(OID_COL)[NATIVE_COL].max().to_dict()
    org_is_pon = oid_df.groupby(OID_COL)[ORG_IS_PON_COL].max().to_dict()

    funding = {}
    for geo, col in FUNDING_COLS.items():
        funding[geo] = oid_df.groupby(OID_COL)[col].max().to_dict()

    locality = {}
    for geo, info in GEOGRAPHIES.items():
        if info["geo_type"] == "reservation":
            locality[geo] = oid_df.groupby(OID_COL)[info["reservation_col"]].max().to_dict()
        else:
            t = oid_df[[OID_COL, info["landscape_col"], COUNTY_NAME_COL, STATE_NAME_COL]].copy()
            t["_local"] = (
                (t[info["landscape_col"]] == 1)
                & t[COUNTY_NAME_COL].str.casefold().eq(info["county_name"].casefold())
                & t[STATE_NAME_COL].str.casefold().eq(info["state_name"].casefold())
            ).astype(int)
            locality[geo] = t.groupby(OID_COL)["_local"].max().to_dict()

    # PON/OID identity crosswalk: exact normalized name first, then controlled prefix partial.
    pon_cids = sorted({c for c in cid_df[CID_COL] if c.startswith("C")})
    pon_name_norm = {c: normalize_name(network_name.get(c, "")) for c in pon_cids}
    eligible_oids = sorted([o for o, v in org_is_pon.items() if v == 1])

    oid_to_pon = {}
    match_type = {}

    for oid in eligible_oids:
        on = normalize_name(org_name.get(oid, ""))
        if not on:
            continue
        exact = [c for c, cn in pon_name_norm.items() if cn and cn == on]
        if len(exact) == 1:
            oid_to_pon[oid] = exact[0]
            match_type[oid] = "Exact"

    for oid in eligible_oids:
        if oid in oid_to_pon:
            continue
        on = normalize_name(org_name.get(oid, ""))
        if len(on.split()) < 3:
            continue
        candidates = [c for c, cn in pon_name_norm.items() if cn.startswith(on + " ")]
        if len(candidates) == 1:
            oid_to_pon[oid] = candidates[0]
            match_type[oid] = "Partial"

    pon_to_oid = {cid: oid for oid, cid in oid_to_pon.items()}

    # Unique source memberships.
    memberships = (
        df[(df[OID_COL] != "") & (df[CID_COL] != "")]
        [[OID_COL, CID_COL]]
        .drop_duplicates()
        .copy()
    )

    # CID scope values can occur on multiple rows.
    cid_scopes = defaultdict(set)
    for _, r in cid_df[[CID_COL, NETWORK_SCOPE_INTEREST_COL]].drop_duplicates().iterrows():
        cid_scopes[r[CID_COL]].update(parse_scope_labels(r[NETWORK_SCOPE_INTEREST_COL]))

    return {
        "df": df,
        "memberships": memberships,
        "org_name": org_name,
        "network_name": network_name,
        "native": native_lookup,
        "org_is_pon": org_is_pon,
        "funding": funding,
        "locality": locality,
        "oid_to_pon": oid_to_pon,
        "pon_to_oid": pon_to_oid,
        "match_type": match_type,
        "cid_scopes": dict(cid_scopes),
    }


def scoped_cids(data, geography, network_type):
    wanted = {x.casefold() for x in GEOGRAPHIES[geography]["scope_labels"]}
    prefix = "C" if network_type == "PON" else "P"
    return {
        cid for cid, scopes in data["cid_scopes"].items()
        if cid.startswith(prefix) and bool(scopes & wanted)
    }


def build_analysis(data, geography, network_type):
    cids = scoped_cids(data, geography, network_type)
    edges_df = data["memberships"][data["memberships"][CID_COL].isin(cids)].copy()

    # Metric graph: retain OIDs as organizations for centrality / projection.
    metric = nx.Graph()
    oid_nodes = sorted(set(edges_df[OID_COL]))
    metric.add_nodes_from(oid_nodes, bipartite="org")
    metric.add_nodes_from(cids, bipartite="cid")
    metric.add_edges_from(edges_df[[OID_COL, CID_COL]].itertuples(index=False, name=None))

    membership_count = {oid: metric.degree(oid) for oid in oid_nodes}
    total_cids = len(cids)
    degree_centrality = {
        oid: (membership_count[oid] / total_cids if total_cids else 0.0)
        for oid in oid_nodes
    }

    if oid_nodes:
        projection = bipartite.weighted_projected_graph(metric, oid_nodes)
    else:
        projection = nx.Graph()

    bridging = set(nx.articulation_points(projection)) if projection.number_of_edges() else set()
    constraint = safe_constraint(projection)

    # Visual graph.
    visual = nx.Graph()

    for oid, cid in edges_df[[OID_COL, CID_COL]].itertuples(index=False, name=None):
        if network_type == "PON" and oid in data["oid_to_pon"]:
            source = data["oid_to_pon"][oid]
        else:
            source = oid

        target = cid
        if source == target:
            continue

        source_is_pon_org = oid in data["oid_to_pon"]
        edge_kind = "PON-to-PON" if network_type == "PON" and source_is_pon_org else "Membership"
        visual.add_edge(source, target, edge_kind=edge_kind, source_oid=oid)

    # Include scoped CID nodes even if no membership survives.
    visual.add_nodes_from(cids)

    return {
        "cids": cids,
        "edges_df": edges_df,
        "metric": metric,
        "projection": projection,
        "visual": visual,
        "oid_nodes": oid_nodes,
        "membership_count": membership_count,
        "degree_centrality": degree_centrality,
        "bridging": bridging,
        "constraint": constraint,
    }


def org_category(data, geography, oid):
    local = data["locality"][geography].get(oid, 0) == 1
    native = data["native"].get(oid, 0) == 1
    if native and local:
        return "Local Native American / Affiliated Organization", LOCAL_NATIVE_COLOR
    if native and not local:
        return "Non-local Native American / Affiliated Organization", NONLOCAL_NATIVE_COLOR
    if local:
        return "Other Local Organization", GEOGRAPHIES[geography]["dark"]
    return "Other Non-local Organization", GEOGRAPHIES[geography]["light"]


def pon_org_category(data, geography, oid):
    local = data["locality"][geography].get(oid, 0) == 1
    if local:
        return "Local PON Organization", LOCAL_PON_ORG_COLOR
    return "Non-local PON Organization", NONLOCAL_PON_ORG_COLOR


def all_six_funding(data, oid):
    return {geo: data["funding"][geo].get(oid, 0) for geo in GEOGRAPHIES}


def node_payload(data, geography, network_type, node_id, pon_analysis, plan_analysis):
    is_oid = node_id in data["org_name"]
    is_cid = node_id in data["network_name"] or node_id.startswith(("C", "P"))

    # PON square may represent a matched OID.
    associated_oid = data["pon_to_oid"].get(node_id, "")
    oid = node_id if is_oid else associated_oid

    if oid:
        local = data["locality"][geography].get(oid, 0) == 1
        is_pon_org = oid in data["oid_to_pon"]
        category, _ = pon_org_category(data, geography, oid) if is_pon_org else org_category(data, geography, oid)

        return {
            "kind": "Organization" if node_id == oid else "PON Organization",
            "id": node_id,
            "oid": oid,
            "name": data["org_name"].get(oid, data["network_name"].get(node_id, node_id)),
            "category": category,
            "local": "Yes" if local else "No",
            "native": "Yes" if data["native"].get(oid, 0) == 1 else "No",
            "is_pon_org": "Yes" if is_pon_org else "No",
            "associated_pon": data["oid_to_pon"].get(oid, ""),
            "pon_bridge": "Yes" if oid in pon_analysis["bridging"] else "No",
            "plan_bridge": "Yes" if oid in plan_analysis["bridging"] else "No",
            "pon_centrality": pon_analysis["degree_centrality"].get(oid, 0.0),
            "plan_centrality": plan_analysis["degree_centrality"].get(oid, 0.0),
            "pon_constraint": pon_analysis["constraint"].get(oid, float("nan")),
            "plan_constraint": plan_analysis["constraint"].get(oid, float("nan")),
            "funding": all_six_funding(data, oid),
        }

    return {
        "kind": "PON" if node_id.startswith("C") else "Plan",
        "id": node_id,
        "name": data["network_name"].get(node_id, node_id),
        "connected_orgs": len(set(
            (pon_analysis if node_id.startswith("C") else plan_analysis)["edges_df"]
            .loc[lambda x: x[CID_COL].eq(node_id), OID_COL]
        )),
    }


def build_vis_html(data, geography, network_type, analysis, pon_analysis, plan_analysis,
                   show_labels, node_scale, edge_scale):
    import json

    vis_nodes = []
    vis_edges = []

    for node_id in analysis["visual"].nodes():
        associated_oid = data["pon_to_oid"].get(node_id, "")

        if node_id.startswith("C"):
            if associated_oid:
                category, color = pon_org_category(data, geography, associated_oid)
            else:
                category, color = "PON", REGULAR_PON_COLOR
            label = data["network_name"].get(node_id, node_id)
            shape = "square"
            connected = analysis["visual"].degree(node_id)
            size = 20 + min(35, connected * 2.2)

        elif node_id.startswith("P"):
            category, color = "Plan", PLAN_COLOR
            label = data["network_name"].get(node_id, node_id)
            shape = "diamond"
            connected = analysis["visual"].degree(node_id)
            size = 20 + min(35, connected * 2.2)

        else:
            oid = node_id
            if oid in data["oid_to_pon"]:
                category, color = pon_org_category(data, geography, oid)
            else:
                category, color = org_category(data, geography, oid)
            label = data["org_name"].get(oid, oid)
            shape = "dot"
            size = 16 + min(28, analysis["membership_count"].get(oid, 0) * 4)

        payload = node_payload(
            data, geography, network_type, node_id, pon_analysis, plan_analysis
        )

        if payload.get("oid"):
            oid = payload["oid"]
            funding_lines = "<br>".join(
                f"<b>{g}:</b> {money(a)}" for g, a in payload["funding"].items()
            )
            title = (
                f"<div style='max-width:420px'>"
                f"<b>{payload['name']}</b><br>"
                f"OID: {payload['oid']}<br>"
                f"Category: {payload['category']}<br>"
                f"Local in {geography}: {payload['local']}<br>"
                f"PON Organization: {payload['is_pon_org']}<br><br>"
                f"<b>PON Bridging Actor:</b> {payload['pon_bridge']}<br>"
                f"<b>Plan Bridging Actor:</b> {payload['plan_bridge']}<br>"
                f"PON Degree Centrality: {payload['pon_centrality']:.4f}<br>"
                f"Plan Degree Centrality: {payload['plan_centrality']:.4f}<br><br>"
                f"<b>Funding for work since 2019</b><br>{funding_lines}"
                f"</div>"
            )
        else:
            title = (
                f"<b>{payload['name']}</b><br>"
                f"Type: {payload['kind']}<br>"
                f"CID: {payload['id']}<br>"
                f"Connected organizations: {payload.get('connected_orgs', 0):,}"
            )

        vis_nodes.append({
            "id": node_id,
            "label": label if show_labels else "",
            "title": title,
            "shape": shape,
            "size": round(size * node_scale, 2),
            "color": {
                "background": color,
                "border": "#333333",
                "highlight": {"background": color, "border": "#111111"},
                "hover": {"background": color, "border": "#111111"},
            },
            "borderWidth": 1.5,
            "font": {"size": 15, "face": "Arial", "color": "#222222"},
        })

    for u, v, attrs in analysis["visual"].edges(data=True):
        pon_to_pon = attrs.get("edge_kind") == "PON-to-PON"
        vis_edges.append({
            "from": u,
            "to": v,
            "color": PON_TO_PON_EDGE_COLOR if pon_to_pon else STANDARD_EDGE_COLOR,
            "width": round((4.5 if pon_to_pon else 2.5) * edge_scale, 2),
            "title": "PON-to-PON tie" if pon_to_pon else "Membership tie",
        })

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    background: white;
    font-family: Arial, sans-serif;
  }}
  #toolbar {{
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }}
  #search {{
    min-width: 300px;
    padding: 8px 10px;
    border: 1px solid #c9c9c9;
    border-radius: 6px;
    font-size: 14px;
  }}
  button {{
    padding: 8px 11px;
    border: 1px solid #c9c9c9;
    background: white;
    border-radius: 6px;
    cursor: pointer;
  }}
  #network {{
    width: 100%;
    height: 760px;
    border: 1px solid #e3e3e3;
    border-radius: 8px;
    background: #ffffff;
  }}
  #hint {{
    color: #666;
    font-size: 12px;
    margin-left: 4px;
  }}
</style>
</head>
<body>
<div id="toolbar">
  <input id="search" type="text" placeholder="Search organization, PON, Plan, OID, or CID">
  <button onclick="findNode()">Find</button>
  <button onclick="network.fit({{animation:true}})">Fit network</button>
  <button onclick="network.stabilize()">Re-layout</button>
  <span id="hint">Scroll to zoom · drag background to pan · drag nodes to reposition · hover for details</span>
</div>
<div id="network"></div>

<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const container = document.getElementById("network");

const data = {{nodes: nodes, edges: edges}};
const options = {{
  autoResize: true,
  interaction: {{
    hover: true,
    tooltipDelay: 150,
    navigationButtons: true,
    keyboard: true,
    multiselect: false
  }},
  physics: {{
    enabled: true,
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{
      gravitationalConstant: -65,
      centralGravity: 0.01,
      springLength: 145,
      springConstant: 0.06,
      damping: 0.55,
      avoidOverlap: 0.65
    }},
    stabilization: {{
      enabled: true,
      iterations: 700,
      updateInterval: 50,
      fit: true
    }}
  }},
  edges: {{
    smooth: {{
      enabled: true,
      type: "continuous",
      roundness: 0.15
    }}
  }}
}};

const network = new vis.Network(container, data, options);

network.once("stabilizationIterationsDone", function () {{
  network.setOptions({{physics: false}});
  network.fit({{animation: {{duration: 500}}}});
}});

function findNode() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  if (!q) return;
  const all = nodes.get();
  const hit = all.find(n =>
    String(n.id).toLowerCase().includes(q) ||
    String(n.label || "").toLowerCase().includes(q)
  );
  if (hit) {{
    network.selectNodes([hit.id]);
    network.focus(hit.id, {{
      scale: 1.35,
      animation: {{duration: 650, easingFunction: "easeInOutQuad"}}
    }});
  }} else {{
    alert("No matching node found.");
  }}
}}

document.getElementById("search").addEventListener("keydown", function(e) {{
  if (e.key === "Enter") findNode();
}});
</script>
</body>
</html>
"""
    return html

def combined_org_table(data, geography, pon, plan):
    all_oids = sorted(set(pon["oid_nodes"]) | set(plan["oid_nodes"]))
    rows = []

    for oid in all_oids:
        is_pon_org = oid in data["oid_to_pon"]
        category, _ = pon_org_category(data, geography, oid) if is_pon_org else org_category(data, geography, oid)
        row = {
            "OID": oid,
            "Organization": data["org_name"].get(oid, ""),
            "Local Status": "Local" if data["locality"][geography].get(oid, 0) == 1 else "Non-local",
            "Organization Category": category,
            "Is PON Organization": "Yes" if is_pon_org else "No",
            "PON Bridging Actor": "Yes" if oid in pon["bridging"] else "No",
            "Plan Bridging Actor": "Yes" if oid in plan["bridging"] else "No",
            "PON Degree Centrality": pon["degree_centrality"].get(oid, 0.0),
            "Plan Degree Centrality": plan["degree_centrality"].get(oid, 0.0),
            "PON Burt Constraint": pon["constraint"].get(oid, float("nan")),
            "Plan Burt Constraint": plan["constraint"].get(oid, float("nan")),
        }
        for geo in GEOGRAPHIES:
            row[f"Funding — {geo}"] = data["funding"][geo].get(oid, 0)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# APP
# ============================================================

st.title("Interactive PON / Plan Social Network Analysis")
st.caption(
    "Explore each county and reservation separately. Zoom, pan, drag nodes, "
    "click a node for details, and inspect PON and Plan bridging actors separately."
)

DATA_FILE = Path(__file__).with_name("MASTER dyads.xlsx")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Optional: temporarily use a newer MASTER dyads.xlsx",
        type=["xlsx"],
        help="If no file is uploaded here, the deployed app uses the bundled MASTER dyads.xlsx."
    )

try:
    if uploaded is not None:
        workbook_bytes = uploaded.getvalue()
        data_source_label = "Uploaded workbook"
    elif DATA_FILE.exists():
        workbook_bytes = DATA_FILE.read_bytes()
        data_source_label = "Bundled MASTER dyads.xlsx"
    else:
        st.error(
            "MASTER dyads.xlsx is not present in the deployed app. "
            "Add it to the same repository folder as app.py, or upload it in the sidebar."
        )
        st.stop()

    data = prepare_data(workbook_bytes)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Data source: {data_source_label}")

controls, graph_col = st.columns([1, 4], gap="large")

with controls:
    st.subheader("Network controls")
    geography = st.selectbox("County / Reservation", list(GEOGRAPHIES.keys()))
    network_type = st.radio("Network type", ["PON", "Plan"], horizontal=True)
    show_labels = st.checkbox("Show node labels", value=True)
    node_scale = st.slider("Node size", 0.6, 2.0, 1.0, 0.1)
    edge_scale = st.slider("Tie thickness", 0.5, 2.0, 1.0, 0.1)

pon = build_analysis(data, geography, "PON")
plan = build_analysis(data, geography, "Plan")
current = pon if network_type == "PON" else plan

graph_html = build_vis_html(
    data, geography, network_type, current, pon, plan,
    show_labels, node_scale, edge_scale
)

with graph_col:
    st.subheader(f"{geography} — {network_type} Network")
    st.caption(
        f"{len(current['visual'].nodes()):,} displayed nodes · "
        f"{len(current['visual'].edges()):,} displayed ties · "
        f"{len(current['bridging']):,} {network_type} bridging actors"
    )
    components.html(graph_html, height=835, scrolling=False)

selected = None

# Node details
st.divider()
st.subheader("Organization details")

detail_oids = sorted(
    set(pon["oid_nodes"]) | set(plan["oid_nodes"]),
    key=lambda oid: data["org_name"].get(oid, oid).casefold()
)

detail_options = ["— Select an organization —"] + [
    f"{data['org_name'].get(oid, oid)} [{oid}]" for oid in detail_oids
]
detail_map = {
    f"{data['org_name'].get(oid, oid)} [{oid}]": oid for oid in detail_oids
}

chosen_label = st.selectbox(
    "Search/select an organization for full metrics and funding",
    detail_options,
)

if chosen_label != "— Select an organization —":
    oid = detail_map[chosen_label]
    payload = node_payload(data, geography, network_type, oid, pon, plan)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {payload['name']}")
        st.write(f"**OID:** {payload['oid']}")
        st.write(f"**Category:** {payload['category']}")
        st.write(f"**Local in {geography}:** {payload['local']}")
        st.write(f"**Native American / Affiliated:** {payload['native']}")
        st.write(f"**PON Organization:** {payload['is_pon_org']}")
        if payload["associated_pon"]:
            st.write(f"**Associated PON:** {payload['associated_pon']}")

    with c2:
        st.markdown("### Network roles")
        st.metric("PON Bridging Actor", payload["pon_bridge"])
        st.metric("Plan Bridging Actor", payload["plan_bridge"])
        st.write(f"**PON Degree Centrality:** {payload['pon_centrality']:.4f}")
        st.write(f"**Plan Degree Centrality:** {payload['plan_centrality']:.4f}")
        pc = payload["pon_constraint"]
        plc = payload["plan_constraint"]
        st.write(f"**PON Burt Constraint:** {'—' if pd.isna(pc) else f'{pc:.4f}'}")
        st.write(f"**Plan Burt Constraint:** {'—' if pd.isna(plc) else f'{plc:.4f}'}")

    with c3:
        st.markdown("### Funding for work since 2019")
        for geo, amount in payload["funding"].items():
            st.write(f"**{geo}:** {money(amount)}")
else:
    st.info(
        "Hover over nodes in the graph for quick details, or select an organization "
        "here for complete PON/Plan metrics and all six funding amounts."
    )


st.markdown("### Network legend")
legend_cols = st.columns(4)
with legend_cols[0]:
    st.markdown("● **Organization** — circle")
    st.markdown(f"<span style='color:{LOCAL_NATIVE_COLOR};font-size:22px'>●</span> Local Native American / Affiliated", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{NONLOCAL_NATIVE_COLOR};font-size:22px'>●</span> Non-local Native American / Affiliated", unsafe_allow_html=True)
with legend_cols[1]:
    st.markdown(f"<span style='color:{LOCAL_PON_ORG_COLOR};font-size:22px'>■</span> Local PON Organization", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{NONLOCAL_PON_ORG_COLOR};font-size:22px'>■</span> Non-local PON Organization", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{REGULAR_PON_COLOR};font-size:22px'>■</span> PON", unsafe_allow_html=True)
with legend_cols[2]:
    st.markdown(f"<span style='color:{PLAN_COLOR};font-size:22px'>◆</span> Plan", unsafe_allow_html=True)
    st.markdown("County/reservation colors distinguish other local vs. non-local organizations.")
with legend_cols[3]:
    st.markdown(f"<span style='color:{STANDARD_EDGE_COLOR};font-size:22px'>━</span> Membership tie", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{PON_TO_PON_EDGE_COLOR};font-size:22px'>━</span> PON-to-PON tie", unsafe_allow_html=True)

# Tables
st.divider()
st.subheader(f"{geography} organization tables")

table = combined_org_table(data, geography, pon, plan)

tab_all, tab_pon_bridge, tab_plan_bridge, tab_funding = st.tabs(
    ["All Organizations", "PON Bridging Actors", "Plan Bridging Actors", "Funding"]
)

with tab_all:
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PON Degree Centrality": st.column_config.NumberColumn(format="%.4f"),
            "Plan Degree Centrality": st.column_config.NumberColumn(format="%.4f"),
            "PON Burt Constraint": st.column_config.NumberColumn(format="%.4f"),
            "Plan Burt Constraint": st.column_config.NumberColumn(format="%.4f"),
        },
    )

with tab_pon_bridge:
    st.dataframe(
        table[table["PON Bridging Actor"] == "Yes"],
        use_container_width=True,
        hide_index=True,
    )

with tab_plan_bridge:
    st.dataframe(
        table[table["Plan Bridging Actor"] == "Yes"],
        use_container_width=True,
        hide_index=True,
    )

with tab_funding:
    funding_cols = ["OID", "Organization"] + [f"Funding — {g}" for g in GEOGRAPHIES]
    st.dataframe(
        table[funding_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            f"Funding — {g}": st.column_config.NumberColumn(format="$%,.0f")
            for g in GEOGRAPHIES
        },
    )

# Downloads
st.divider()
csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download {geography} organization table (CSV)",
    data=csv,
    file_name=f"{geography.replace(' ', '_')}_organization_metrics.csv",
    mime="text/csv",
)

with st.expander("Network definitions used by this app"):
    st.markdown(
        """
- **PON and Plan are analyzed separately.**
- **PON Bridging Actor** is an articulation point in the PON organization projection.
- **Plan Bridging Actor** is an articulation point in the Plan organization projection.
- Degree centrality is direct PON/Plan membership count divided by the total scoped PONs/Plans.
- The organization projection is used for Burt constraint and bridging actors, not for degree centrality.
- Matched PON/OID identities collapse to the PON square on PON maps; Plan maps retain organizations as circles.
- County locality uses the appropriate landscape flag plus county and state.
- Reservation locality uses only the corresponding reservation located-in flag.
- Funding is displayed independently for all six county/reservation funding fields.
        """
    )
