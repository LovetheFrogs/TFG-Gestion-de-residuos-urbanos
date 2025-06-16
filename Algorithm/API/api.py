from flask import Flask, jsonify, request
import base64
import sys
import os
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Algorithm.Library.problem import model, algorithms
import psycopg2
from psycopg2.extras import Json
import pickle

app = Flask(__name__)


def encode_img(path):
    with open(path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode('ascii')
        
        return encoded


@app.route("/load", methods=["POST"])
def load():
    """Loads the graph from the database. Needs to be logged in. Returns an image of the loaded graph."""
    data = request.get_json()
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON or missing request body."
        }), 400
    
    if "auth" not in data or "conn" not in data:
        return jsonify({
            "status": "error",
            "message": "Malformed request body."
        }), 400
    
    auth = bool(data["auth"])
    db_conn = dict(data["conn"])
    if not auth:
        return jsonify({
            "status": "Unathorised",
            "message": "Session not authorised. Please, login to the database."
        }), 401
        
    try:
        graph = model.Graph()
        conn = psycopg2.connect(
            dbname = db_conn["dbname"],
            user = db_conn["user"],
            password = db_conn["password"],
            host = db_conn["host"],
            port = db_conn["port"]
        )
        cur = conn.cursor()
        graph.populate_from_database(cur=cur)
        cur.close()
        conn.close()
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to load graph: {str(e)}"
        }), 500
        
    return jsonify({
        "status": "success",
        "message": "Graph loaded successfully.",
        "data" : base64.b64encode(pickle.dumps(graph)).decode('utf-8')
    }), 200
    
    
@app.route("/divide", methods=["POST"])
def divide():
    """Divides the graph into zones. Truck weight must be indicated in the JSON request. Must provide a graph. Returns a list of the zones and an image of it."""
    data = request.get_json()
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON or missing request body."
        }), 400
        
    if "capacity" not in data or "graph" not in data:
        return jsonify({
            "status": "error",
            "message": "Malformed request body."
        }), 400
    
    capacity = float(data["capacity"])
    decoded_graph = base64.b64decode(data["graph"])
    graph = pickle.load(io.BytesIO(decoded_graph))
    
    try:
        algo = algorithms.Algorithms(graph)
        _, zones = algo.divide(capacity, dir="False")
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Graph division failed: {str(e)}"
        }), 500
    
    zones_int = []
    for z in zones:
        aux = []
        for n in z:
            aux.append(n.index)
        zones_int.append(aux)
  
    return jsonify({
        "status": "success",
        "message": f"Graph divided into {len(zones_int)} zones successfully.",
        "zones": zones_int
    }), 200
    

@app.route("/solve", methods=["POST"])
def solve():
    """Uses the run function to solve a graph. A graph must be provided. Returns the route, value and image of it."""
    data = request.get_json()
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON or missing request body."
        }), 400
        
    if "graph" not in data:
        return jsonify({
            "status": "error",
            "message": "Malformed request body."
        }), 400

    decoded_graph = base64.b64decode(data["graph"])
    graph = pickle.load(io.BytesIO(decoded_graph))
    
    try:
        algo = algorithms.Algorithms(graph)
        route, val = algo.run(dir="False")
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to find route: {str(e)}"
        }), 500
        
    return jsonify({
        "status": "success",
        "message": "Successfully found a route.",
        "route": route,
        "value": val
    }), 200
            

@app.route("/run", methods=["POST"])
def run():
    """Divides a graph into zones and solves each of them. Returns a JSON with a list of routes, a list of values, the total value, a list of all images, the final image with all zones."""
    data = request.get_json()
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON or missing request body."
        }), 400
        
    if "capacity" not in data or "graph" not in data:
        return jsonify({
            "status": "error",
            "message": "Malformed request body."
        }), 400
    
    capacity = float(data["capacity"])
    decoded_graph = base64.b64decode(data["graph"])
    graph = pickle.load(io.BytesIO(decoded_graph))
    
    try:
        algo = algorithms.Algorithms(graph)
        subgraphs, _ = algo.divide(zone_weight=capacity, dir="False")
        results, values, points = [], [], []
        print(f"Dividing in {len(subgraphs)} zones")
        for i, sg in enumerate(subgraphs):
            print(f"Computing route {i}, {sg.nodes} nodes")
            algo = algorithms.Algorithms(sg)
            route, val = algo.run(dir="False", vrb=True)
            results.append(route)
            values.append(val)
            points.append([sg.get_node(n).coordinates for n in route])
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to run algorithm: {str(e)}"
        }), 500
        
    return jsonify({
        "status": "success",
        "message": "Successfully ran the algorithm.",
        "routes": results,
        "values": values,
        "map": base64.b64encode(pickle.dumps(points)).decode('utf-8')
    }), 200


@app.route("/send-route", methods=["POST"])
def send_routes():
    data = request.get_json()
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON or missing request body."
        }), 400
    
    if "auth" not in data or "db_conn" not in data or "tid" not in data or "route" not in data:
        return jsonify({
            "status": "error",
            "message": "Malformed request body."
        }), 400
    
    auth = bool(data["auth"])
    db_conn = dict(data["db_conn"])
    if not auth:
        return jsonify({
            "status": "Unathorised",
            "message": "Session not authorised. Please, login to the database."
        }), 401
    tid = int(data["tid"])
    route = list(data["route"])
    
    try:
        conn = psycopg2.connect(
            dbname = db_conn["dbname"],
            user = db_conn["user"],
            password = db_conn["password"],
            host = db_conn["host"],
            port = db_conn["port"]
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO PENDING_ROUTES (TRUCK_ID, ROUTE) VALUES (%s, %s)",
            (tid, Json(route))
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({
            "status": "success",
            "message": "The data has been sent successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to send data: {str(e)}"
        }), 500


@app.route("/get-route", methods=["GET"])
def get_route():
    data = request.get_json()
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON or missing request body."
        }), 400
    
    if "tid" not in data:
        return jsonify({
            "status": "error",
            "message": "Malformed request body."
        }), 400

    tid = int(data["tid"])
    
    try:
        conn = psycopg2.connect(
            dbname="logistics",
            user="truck",
            password="truck_db",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT ID, ROUTE FROM PENDING_ROUTES WHERE TRUCK_ID = %s ORDERED BY CREATED ASC LIMIT 1",
            (tid)
        )
        row = cur.fetchone()
        if not row:
            return jsonify({
                "status": "empty",
                "message": "No route was found"
            }), 200
        
        idx, route = row
        
        cur.execute(
            "DELETE FROM PENDING_ROUTES WHERE ID = %s",
            (idx)
        )
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"Route obtained: {str(route)}",
            "route": route
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to load data: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
