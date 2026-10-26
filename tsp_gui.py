pip install flask
import math
import os
import sys
from flask import Flask, request, jsonify, render_template_string
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

app = Flask(__tsp.gui__)


# --- User Provided Functions ---

def read_tsp_file(file_path):
    """
    Read the TSP file, extract coordinates, and compute the edge matrix
    dynamically for EUC_2D or read it directly for EXPLICIT formats.
    """
    with open(file_path, 'r') as file:
        lines = [line.replace(':', '').strip() for line in file.readlines()]

        specifications = {}
        for line in lines:
            if line in ['EDGE_WEIGHT_SECTION', 'NODE_COORD_SECTION',
                        'DISPLAY_DATA_SECTION']:
                break
            item = line.split()
            if len(item) > 1:
                specifications[item[0]] = item[1:]

        num_cities = int(specifications['DIMENSION'][0])
        edge_weight_type = specifications.get('EDGE_WEIGHT_TYPE', ['EXPLICIT'])[
            0]

        coordinates = None
        coord_idx = -1
        if 'NODE_COORD_SECTION' in lines:
            coord_idx = lines.index('NODE_COORD_SECTION')
        elif 'DISPLAY_DATA_SECTION' in lines:
            coord_idx = lines.index('DISPLAY_DATA_SECTION')

        if coord_idx != -1:
            coordinates = {}
            for line in lines[coord_idx + 1:]:
                if line in ['EOF', 'EDGE_WEIGHT_SECTION',
                            'DISPLAY_DATA_SECTION'] or not line:
                    break
                item = line.split()
                if len(item) >= 3:
                    coordinates[int(item[0]) - 1] = (
                        float(item[1]), float(item[2]))

        edge_matrix = [[0] * num_cities for _ in range(num_cities)]

        if edge_weight_type == 'EUC_2D':
            if not coordinates:
                raise ValueError(
                    f"File {file_path} specifies EUC_2D but contains no coordinate section.")
            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        dx = coordinates[i][0] - coordinates[j][0]
                        dy = coordinates[i][1] - coordinates[j][1]
                        edge_matrix[i][j] = int(
                            round(math.sqrt(dx ** 2 + dy ** 2)))
        else:
            index = lines.index('EDGE_WEIGHT_SECTION')
            flat_data = [
                int(value) for line in lines[index + 1:]
                if line and line.strip() not in ['DISPLAY_DATA_SECTION',
                                                 'NODE_COORD_SECTION', 'EOF']
                for value in line.split() if value.lstrip('-').isdigit()
            ]
            format_type = \
                specifications.get('EDGE_WEIGHT_FORMAT', ['FULL_MATRIX'])[0]
            idx = 0

            if format_type == 'LOWER_DIAG_ROW':
                for i in range(num_cities):
                    for j in range(i + 1):
                        edge_matrix[i][j] = edge_matrix[j][i] = flat_data[idx]
                        idx += 1
            elif format_type == 'UPPER_DIAG_ROW':
                for i in range(num_cities):
                    for j in range(i, num_cities):
                        edge_matrix[i][j] = edge_matrix[j][i] = flat_data[idx]
                        idx += 1
            else:
                for i in range(num_cities):
                    edge_matrix[i] = [flat_data[idx + j] for j in
                                      range(num_cities)]
                    idx += num_cities

    return specifications, edge_matrix, coordinates, num_cities


def tsp_model(edge_matrix, num_cities):
    """ Create a model with degree constraints """
    model = gp.Model('DFJ_TSP_Edge')
    model.setParam('OutputFlag', 0)
    n = num_cities
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
    model.setObjective(
        gp.quicksum(edge_matrix[i][j] * x[i, j] for (i, j) in edges),
        GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(x[min(i, j), max(i, j)] for j in range(n) if i != j) == 2
         for i in range(n)),
        name="Degree_Constraints"
    )

    model.update()
    return model, x, edges


# --- Global State & Initialization ---

TSP_FILE = "./STSP/dantzig42.tsp"
model = None
x = {}
N = 0
edges = []
nodes = {}
edge_matrix = None
action_stack = []


def initialize_system():
    global model, x, N, edges, nodes, edge_matrix, action_stack

    if not os.path.exists(TSP_FILE):
        print(
            f"\n[ERROR] File '{TSP_FILE}' not found in the current working directory.")
        print(
            "Please place the TSPLIB 'dantzig42.tsp' file here and restart the application.\n")
        sys.exit(1)

    specs, edge_matrix, coords, N = read_tsp_file(TSP_FILE)

    if coords is None:
        coords = {i: (50 + 40 * math.cos(2 * math.pi * i / N),
                      50 + 40 * math.sin(2 * math.pi * i / N)) for i in
                  range(N)}

    nodes = coords
    action_stack.clear()
    model, x, edges = tsp_model(edge_matrix, N)


def export_support_graph(filename="support_graph.graphml"):
    if model.Status != GRB.OPTIMAL:
        return None

    G = nx.Graph()
    for i in range(N):
        G.add_node(i, x=float(nodes[i][0]), y=float(nodes[i][1]))

    for (i, j) in edges:
        val = x[i, j].X
        if val > 1e-4:
            G.add_edge(i, j, weight=float(val), cost=int(edge_matrix[i][j]))

    nx.write_graphml(G, filename)
    return filename


def get_solution_state(message=None):
    sol_edges = []
    obj_val = 0.0

    if model.Status == GRB.OPTIMAL:
        obj_val = model.ObjVal
        export_support_graph()

        for (i, j) in edges:
            var = x[i, j]
            if var.X > 1e-4:
                sol_edges.append({
                    "source": i, "target": j, "val": var.X,
                    "x1": nodes[i][0], "y1": nodes[i][1],
                    "x2": nodes[j][0], "y2": nodes[j][1]
                })

    eq_list = [action["eq_str"] for action in action_stack if
               "eq_str" in action]
    eq_list.reverse()

    return {
        "nodes": [{"id": i, "x": nodes[i][0], "y": nodes[i][1]} for i in
                  range(N)],
        "edges": sol_edges,
        "objVal": obj_val,
        "message": message,
        "equations": eq_list
    }


initialize_system()


def format_set(S):
    return "{" + ",".join(map(str, sorted(S))) + "}"


# --- API Endpoints ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/solve', methods=['GET'])
def solve():
    model.optimize()
    return jsonify(get_solution_state())


@app.route('/reset', methods=['POST'])
def reset():
    initialize_system()
    model.optimize()
    return jsonify(
        get_solution_state("System reset to continuous LP relaxation."))


@app.route('/undo', methods=['POST'])
def undo():
    if not action_stack:
        return jsonify(get_solution_state("Nothing to undo."))

    last_action = action_stack.pop()

    if last_action["type"] == "cut":
        for c in last_action["constrs"]:
            model.remove(c)
    elif last_action["type"] == "branch":
        for (i, j) in edges:
            x[i, j].VType = GRB.CONTINUOUS

    model.update()
    model.optimize()
    return jsonify(get_solution_state("Undo successful."))


@app.route('/branch', methods=['POST'])
def branch():
    for (i, j) in edges:
        x[i, j].VType = GRB.BINARY

    model.update()
    model.optimize()

    action_stack.append({
        "type": "branch",
        "eq_str": "<span style='color: #67e8f9;'>❖ Variables converted to BINARY (MILP Mode)</span>"
    })

    return jsonify(get_solution_state("Variable types changed to BINARY."))


@app.route('/export', methods=['GET'])
def export_manual():
    filename = export_support_graph()
    if filename:
        return jsonify({"status": "success",
                        "message": f"Successfully exported to {filename} in your working directory."})
    return jsonify({"status": "error",
                    "message": "No optimal solution available to export."})


@app.route('/export_lp', methods=['GET'])
def export_lp():
    try:
        filename = "tsp_formulation.lp"
        model.write(filename)
        return jsonify({"status": "success",
                        "message": f"Successfully written model to {filename} in your working directory."})
    except Exception as e:
        return jsonify({"status": "error",
                        "message": f"Failed to write LP file: {str(e)}"})


@app.route('/add_sec', methods=['POST'])
def add_sec():
    S = request.json['S']
    if len(S) > 1:
        expr = gp.quicksum(x[i, j] for i in S for j in S if i < j)
        c = model.addConstr(expr <= len(S) - 1, name=f"sec_{model.NumConstrs}")
        model.optimize()

        eq_str = f"<span style='color:#fda4af'>SEC:</span> x({format_set(S)}) &le; {len(S) - 1}"
        action_stack.append({"type": "cut", "constrs": [c], "eq_str": eq_str})

    return jsonify(get_solution_state())


@app.route('/add_2connected', methods=['POST'])
def add_2connected():
    v = request.json['node']

    support = [(i, j) for (i, j) in edges if
               x[i, j].X > 1e-4 and i != v and j != v]
    G = nx.Graph()
    G.add_nodes_from(set(range(N)) - {v})
    G.add_edges_from(support)

    components = list(nx.connected_components(G))

    added_constrs = []
    eq_strs = []
    base_constr_idx = model.NumConstrs

    for loop_idx, comp in enumerate(components):
        if 0 < len(comp) < N - 1:
            expr = gp.quicksum(x[i, j] for i in comp for j in comp if i < j)
            c = model.addConstr(expr <= len(comp) - 1,
                                name=f"2con_sec_{base_constr_idx + loop_idx}")
            added_constrs.append(c)
            eq_strs.append(
                f"<span style='color:#fdba74'>2-Con SEC:</span> x({format_set(comp)}) &le; {len(comp) - 1}")
            break  # Stop after adding the first component

    if added_constrs:
        model.optimize()
        action_stack.append({"type": "cut", "constrs": added_constrs,
                             "eq_str": "<br>".join(eq_strs)})

    return jsonify(get_solution_state())


@app.route('/add_comb', methods=['POST'])
def add_comb():
    data = request.json
    H = data['H']
    teeth = data['T']
    k = len(teeth)

    if k >= 3 and k % 2 != 0:
        expr_H = gp.quicksum(x[i, j] for i in H for j in H if i < j)
        expr_T = gp.quicksum(
            x[min(i, j), max(i, j)] for T in teeth for i in T for j in T if
            i < j)

        # Updated RHS logic: size of handle + size of all teeth - (3*k + 1)/2
        size_handle = len(H)
        size_teeth = sum(len(T) for T in teeth)
        rhs = size_handle + size_teeth - (3 * k + 1) / 2

        # Updated to <= inequality
        c = model.addConstr(expr_H + expr_T <= rhs,
                            name=f"comb_{model.NumConstrs}")
        model.optimize()

        h_str = f"x({format_set(H)})"
        t_strs = [f"x({format_set(T)})" for T in teeth]
        # Match the purple UI color for the comb indicator
        eq_str = f"<span style='color:#d8b4fe'>Comb:</span> {h_str} + {' + '.join(t_strs)} &le; {rhs}"

        action_stack.append({"type": "cut", "constrs": [c], "eq_str": eq_str})

    return jsonify(get_solution_state())


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Symmetric TSP Cut Demonstrator</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --bg-main: #f8fafc;
            --bg-sidebar: #0f172a;
            --bg-card: #1e293b;
            --border-card: #334155;
            --text-muted: #94a3b8;
            --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;

            /* Pastel Tones */
            --clr-primary: #93c5fd; 
            --clr-success: #86efac; 
            --clr-danger: #fda4af; 
            --clr-warning: #fdba74; 
            --clr-purple: #d8b4fe; 
        }

        body { 
            font-family: var(--font-stack); 
            display: flex; 
            margin: 0; 
            height: 100vh; 
            background-color: var(--bg-main);
            color: #0f172a;
            -webkit-font-smoothing: antialiased;
            overflow: hidden;
        }

        #canvas-container { 
            flex: 1; 
            position: relative; 
            background: var(--bg-main);
            cursor: grab;
        }
        #canvas-container:active { cursor: grabbing; }

        #sidebar { 
            width: 360px; 
            padding: 24px; 
            background: var(--bg-sidebar); 
            color: #f8fafc; 
            overflow-y: auto; 
            display: flex; 
            flex-direction: column;
            box-shadow: -4px 0 24px rgba(0,0,0,0.15);
            z-index: 10;
        }

        svg { width: 100%; height: 100%; }

        h3 { font-size: 1.25rem; font-weight: 600; margin: 0 0 4px 0; letter-spacing: -0.025em; }
        .subtitle { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 20px; }
        h4 { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin: 24px 0 12px 0; }

        button { 
            display: block; 
            width: 100%; 
            padding: 10px 14px; 
            margin-bottom: 8px; 
            border: 1px solid transparent;
            border-radius: 6px; 
            cursor: pointer; 
            font-weight: 600; 
            font-size: 0.9rem;
            transition: all 0.15s ease;
            box-sizing: border-box;
        }
        button:hover { opacity: 0.95; transform: translateY(-0.5px); }
        button:active { transform: translateY(0); }
        button:disabled { opacity: 0.25; cursor: not-allowed; transform: none; }

        /* Adjusting text color to dark for pastel backgrounds to maintain legibility */
        .btn-primary { background: var(--clr-primary); color: #0f172a; }
        .btn-success { background: var(--clr-success); color: #0f172a; }
        .btn-danger { background: var(--clr-danger); color: #0f172a; }
        .btn-purple { background: var(--clr-purple); color: #0f172a; }
        .btn-warning { background: var(--clr-warning); color: #0f172a; }
        .btn-secondary { background: #334155; color: #f8fafc; border-color: #475569; }
        .btn-secondary:hover { background: #475569; }

        .btn-outline { background: transparent; border: 1px solid #334155; color: #e2e8f0; font-weight: 500;}
        .btn-outline:hover { background: rgba(255,255,255,0.05); }
        .btn-outline.active { background: white; color: var(--bg-sidebar); border-color: white; }

        .btn-row { display: flex; gap: 8px; margin-bottom: 8px; }
        .btn-row button { margin-bottom: 0; }

        #status-box { 
            background: var(--bg-card); 
            border: 1px solid var(--border-card);
            padding: 14px; 
            border-radius: 8px; 
            margin-bottom: 16px; 
            font-size: 0.88rem; 
            line-height: 1.5;
        }
        #status-lbl { font-weight: 600; color: #fff; }
        #obj-val { font-family: monospace; font-size: 1rem; color: #f8fafc; font-weight: 600; }

        #equation-box { 
            flex: 1; 
            overflow-y: auto; 
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; 
            font-size: 0.8rem;
            background: #090d16;
            border: 1px solid var(--border-card);
            padding: 12px;
            border-radius: 6px;
            color: #e2e8f0;
            line-height: 1.6;
        }

        hr { border: 0; border-top: 1px solid var(--border-card); margin: 20px 0; }

        #comb-controls { 
            display: none; 
            background: #090d16; 
            border: 1px solid var(--border-card);
            padding: 12px; 
            border-radius: 6px; 
            margin-bottom: 12px; 
        }
        .tooth-btn { display: inline-block; width: 48%; margin: 1%; font-size: 0.78rem; padding: 6px 8px; }

        #zoom-controls {
            position: absolute;
            bottom: 24px;
            right: 24px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            z-index: 5;
        }
        .zoom-btn {
            background: white;
            color: #0f172a;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            width: 36px;
            height: 36px;
            padding: 0;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
            font-weight: bold;
        }
        .zoom-btn:hover { background: #f8fafc; transform: none; opacity: 1; }

        .node { stroke-width: 2px; cursor: pointer; transition: stroke-width 0.15s, r 0.15s; }
        .node:hover { stroke-width: 4px; r: 16px; }
        .edge { transition: stroke 0.25s, stroke-width 0.25s; }
        .edge-label { font-size: 10px; fill: #64748b; font-weight: 500; font-family: monospace; }
    </style>
</head>
<body>
    <div id="canvas-container">
        <svg id="graph-svg"></svg>
        <div id="zoom-controls">
            <button class="zoom-btn" onclick="triggerZoom(1.3)" title="Zoom In">＋</button>
            <button class="zoom-btn" onclick="triggerZoom(1/1.3)" title="Zoom Out">－</button>
            <button class="zoom-btn" onclick="resetZoom()" title="Reset Perspective" style="font-size: 0.95rem;">⟲</button>
        </div>
    </div>
    <div id="sidebar">
        <h3>STSP Cut Demonstrator</h3>
        <div class="subtitle">Interactive Formulation Explorer</div>

        <div id="status-box">
            <div><span id="status-lbl">Status:</span> <span id="status-txt" style="color: var(--text-muted)">Ready</span></div>
            <div style="margin-top: 4px;">Objective: <span id="obj-val">-</span></div>
        </div>

        <div class="btn-row">
            <button class="btn-secondary" onclick="resetLP()">Reset LP</button>
            <button class="btn-warning" id="btn-undo" onclick="undoAction()" style="background: rgba(253, 186, 116, 0.15); color: var(--clr-warning); border-color: rgba(253, 186, 116, 0.3);" disabled>Undo Last</button>
        </div>

        <div class="btn-row">
            <button class="btn-outline" onclick="exportGraphML()" style="font-size: 0.8rem; padding: 8px;">Export GraphML</button>
            <button class="btn-outline" onclick="exportLP()" style="font-size: 0.8rem; padding: 8px;">Export .LP</button>
        </div>

        <h4>Add Cutting Planes</h4>
        <button class="btn-secondary" onclick="branchAndSolve()" style="margin-bottom: 12px;">Solve as MIP</button>
        <button class="btn-outline" onclick="startSEC()" style="text-align: left; border-left: 3px solid var(--clr-danger)">1. Connectivity SEC</button>
        <button class="btn-outline" onclick="start2Connected()" style="text-align: left; border-left: 3px solid var(--clr-warning)">2. 2-Connected SEC</button>
        <button class="btn-outline" onclick="startComb()" style="text-align: left; border-left: 3px solid var(--clr-purple)">3. Comb Inequality</button>

        <div id="comb-controls">
            <button id="btn-handle" class="btn-outline active" onclick="setCombActive('handle')" style="border-color: var(--clr-purple);">Select Handle (H)</button>
            <div id="teeth-container" style="margin-bottom: 6px;"></div>
            <button class="btn-outline" onclick="addTooth()" style="border-style: dashed; font-size: 0.8rem; padding: 6px;">+ Add Tooth</button>
        </div>

        <div id="action-controls" style="display: none; margin-top: 8px;">
            <div class="btn-row">
                <button id="btn-submit" class="btn-success" onclick="submitCut()">Submit Cut</button>
                <button class="btn-secondary" onclick="cancelCut()">Cancel</button>
            </div>
        </div>

        <h4>Formulation Hierarchy</h4>
        <div id="equation-box">No constraints added yet.</div>
    </div>

    <script>
        const svg = d3.select("#graph-svg");
        const viewGroup = svg.append("g").attr("class", "view-group");

        let width = document.getElementById('canvas-container').clientWidth;
        let height = document.getElementById('canvas-container').clientHeight;

        let scaleX = d3.scaleLinear();
        let scaleY = d3.scaleLinear();

        const toothColors = ['#fde047', '#86efac', '#fda4af', '#93c5fd', '#94a3b8', '#d8b4fe', '#fbcfe8'];

        let graphData = { nodes: [], edges: [] };
        let mode = 'idle'; 
        let selection = new Set();

        let combData = { handle: new Set(), teeth: [] };
        let combActiveSet = 'handle';

        const zoomBehavior = d3.zoom()
            .scaleExtent([0.1, 12])
            .on("zoom", (event) => {
                viewGroup.attr("transform", event.transform);
            });

        svg.call(zoomBehavior);

        function triggerZoom(factor) {
            svg.transition().duration(220).call(zoomBehavior.scaleBy, factor);
        }

        function resetZoom() {
            svg.transition().duration(250).call(zoomBehavior.transform, d3.zoomIdentity);
        }

        fetch('/solve').then(r => r.json()).then(data => updateGraph(data));

        window.addEventListener('resize', () => {
            width = document.getElementById('canvas-container').clientWidth;
            height = document.getElementById('canvas-container').clientHeight;
            if (graphData.nodes.length > 0) {
                const xExtent = d3.extent(graphData.nodes, d => d.x);
                const yExtent = d3.extent(graphData.nodes, d => d.y);
                const xPad = (xExtent[1] - xExtent[0]) * 0.1 || 10;
                const yPad = (yExtent[1] - yExtent[0]) * 0.1 || 10;
                scaleX.domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([60, width - 60]);
                scaleY.domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([height - 60, 60]);
                renderNetwork();
            }
        });

        function updateGraph(data) {
            graphData = data;
            document.getElementById('obj-val').innerText = data.objVal.toFixed(4);

            if (data.message) {
                setStatus(data.message);
            }

            if (data.equations !== undefined) {
                document.getElementById('equation-box').innerHTML = data.equations.length > 0 
                    ? data.equations.join('<hr style="border-color:#334155; margin:8px 0;">') 
                    : "<span style='color: var(--text-muted)'>No user cuts added yet.</span>";
                document.getElementById('btn-undo').disabled = (data.equations.length === 0);
            }

            const xExtent = d3.extent(graphData.nodes, d => d.x);
            const yExtent = d3.extent(graphData.nodes, d => d.y);
            const xPad = (xExtent[1] - xExtent[0]) * 0.1 || 10;
            const yPad = (yExtent[1] - yExtent[0]) * 0.1 || 10;

            scaleX.domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([60, width - 60]);
            scaleY.domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([height - 60, 60]);

            cancelCut();
            renderNetwork();
        }

        function setStatus(text) {
            document.getElementById('status-txt').innerText = text ? text : "Ready";
        }

        function renderNetwork() {
            viewGroup.selectAll("*").remove();

            viewGroup.selectAll(".edge")
                .data(graphData.edges)
                .enter().append("line")
                .attr("class", "edge")
                .attr("x1", d => scaleX(d.x1))
                .attr("y1", d => scaleY(d.y1))
                .attr("x2", d => scaleX(d.x2))
                .attr("y2", d => scaleY(d.y2))
                .style("stroke", d => d.val >= 0.99 ? "#1e293b" : "#93c5fd")
                .style("stroke-width", d => d.val * 3.5)
                .style("stroke-dasharray", d => d.val < 0.99 ? "4,3" : "none")
                .style("opacity", d => d.val * 0.85 + 0.15);

            viewGroup.selectAll(".edge-label")
                .data(graphData.edges.filter(d => d.val > 0.01 && d.val < 0.99))
                .enter().append("text")
                .attr("class", "edge-label")
                .attr("x", d => scaleX((d.x1 + d.x2) / 2))
                .attr("y", d => scaleY((d.y1 + d.y2) / 2) - 6)
                .attr("text-anchor", "middle")
                .text(d => d.val.toFixed(2));

            viewGroup.selectAll(".node")
                .data(graphData.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("id", d => "node-" + d.id)
                .attr("cx", d => scaleX(d.x))
                .attr("cy", d => scaleY(d.y))
                .attr("r", 13)
                .style("stroke-width", "2px")
                .on("click", (event, d) => handleNodeClick(d.id));

            viewGroup.selectAll(".node-label")
                .data(graphData.nodes)
                .enter().append("text")
                .attr("x", d => scaleX(d.x))
                .attr("y", d => scaleY(d.y) + 3.5)
                .attr("text-anchor", "middle")
                .style("fill", "#0f172a")
                .style("font-size", "10px")
                .style("font-weight", "600")
                .style("pointer-events", "none")
                .text(d => d.id);

            renderNodeColors();
        }

        function renderNodeColors() {
            viewGroup.selectAll(".node").each(function(d) {
                let fill = "#cbd5e1"; 
                let stroke = "#fff";

                if (mode === 'sec' && selection.has(d.id)) {
                    fill = "#fda4af";
                } else if (mode === '2connected' && selection.has(d.id)) {
                    fill = "#fdba74";
                } else if (mode === 'comb') {
                    if (combData.handle.has(d.id)) fill = "#d8b4fe";

                    for (let i = combData.teeth.length - 1; i >= 0; i--) {
                        if (combData.teeth[i].has(d.id)) {
                            stroke = toothColors[i % toothColors.length];
                            break;
                        }
                    }
                }

                d3.select(this)
                  .style("fill", fill)
                  .style("stroke", stroke);
            });
        }

        function handleNodeClick(id) {
            if (mode === 'idle') return;

            if (mode === 'sec' || mode === '2connected') {
                if (mode === '2connected') {
                    const wasSelected = selection.has(id);
                    selection.clear();
                    if (!wasSelected) selection.add(id);
                } else {
                    if (selection.has(id)) selection.delete(id);
                    else selection.add(id);
                }
            } 
            else if (mode === 'comb') {
                if (combActiveSet === 'handle') {
                    if (combData.handle.has(id)) combData.handle.delete(id);
                    else combData.handle.add(id);
                } else if (combActiveSet.startsWith('tooth_')) {
                    let idx = parseInt(combActiveSet.split('_')[1]);
                    if (combData.teeth[idx].has(id)) combData.teeth[idx].delete(id);
                    else combData.teeth[idx].add(id);
                }
            }
            renderNodeColors();
        }

        function cancelCut() {
            if(mode !== 'idle') setStatus("Ready");
            mode = 'idle';
            selection.clear();
            combData = { handle: new Set(), teeth: [] };
            document.getElementById('action-controls').style.display = 'none';
            document.getElementById('comb-controls').style.display = 'none';
            renderNodeColors();
        }

        function resetLP() {
            setStatus("Resetting environment...");
            fetch('/reset', {method: 'POST'}).then(r => r.json()).then(updateGraph);
        }

        function undoAction() {
            setStatus("Undoing last action...");
            fetch('/undo', {method: 'POST'}).then(r => r.json()).then(updateGraph);
        }

        function exportGraphML() {
            fetch('/export').then(r => r.json()).then(data => {
                alert(data.message);
            });
        }

        function exportLP() {
            fetch('/export_lp').then(r => r.json()).then(data => {
                alert(data.message);
            });
        }

        function branchAndSolve() {
            setStatus("Solving MILP Formulation...");
            fetch('/branch', {method: 'POST'}).then(r => r.json()).then(updateGraph);
        }

        function startSEC() {
            cancelCut();
            mode = 'sec';
            setStatus("SEC Mode: Click nodes to map subset S.");
            document.getElementById('action-controls').style.display = 'block';
        }

        function start2Connected() {
            cancelCut();
            mode = '2connected';
            setStatus("2-Con Mode: Target an articulation node.");
            document.getElementById('action-controls').style.display = 'block';
        }

        function startComb() {
            cancelCut();
            mode = 'comb';
            combData = { handle: new Set(), teeth: [] };
            document.getElementById('teeth-container').innerHTML = '';
            document.getElementById('comb-controls').style.display = 'block';
            document.getElementById('action-controls').style.display = 'block';
            setCombActive('handle');
        }

        function addTooth() {
            let idx = combData.teeth.length;
            combData.teeth.push(new Set());

            let color = toothColors[idx % toothColors.length];
            let btn = document.createElement('button');
            btn.className = 'btn-outline tooth-btn';
            btn.id = 'btn-tooth-' + idx;
            btn.style.borderColor = color;
            btn.innerText = `Tooth ${idx + 1}`;
            btn.onclick = () => setCombActive('tooth_' + idx);

            document.getElementById('teeth-container').appendChild(btn);
            setCombActive('tooth_' + idx);
        }

        function setCombActive(target) {
            combActiveSet = target;

            document.getElementById('btn-handle').classList.remove('active');
            document.querySelectorAll('.tooth-btn').forEach(b => b.classList.remove('active'));

            if (target === 'handle') {
                document.getElementById('btn-handle').classList.add('active');
                setStatus("Comb: Map out the structural Handle (H).");
            } else {
                let idx = target.split('_')[1];
                document.getElementById('btn-tooth-' + idx).classList.add('active');
                setStatus(`Comb: Selecting constituents for Tooth ${parseInt(idx)+1}.`);
            }
        }

        function submitCut() {
            let endpoint, payload;

            if (mode === 'sec') {
                if (selection.size < 2) return alert("Select at least 2 nodes.");
                endpoint = '/add_sec';
                payload = { S: Array.from(selection) };
            } else if (mode === '2connected') {
                if (selection.size !== 1) return alert("Select exactly one node.");
                endpoint = '/add_2connected';
                payload = { node: Array.from(selection)[0] };
            } else if (mode === 'comb') {
                if (combData.teeth.length < 3) return alert("Need at least 3 teeth.");
                endpoint = '/add_comb';
                payload = { 
                    H: Array.from(combData.handle), 
                    T: combData.teeth.map(t => Array.from(t))
                };
            }

            fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(updateGraph);
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)
