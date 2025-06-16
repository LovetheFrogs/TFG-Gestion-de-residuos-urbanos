from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import sys
import os
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Algorithm.Library.problem import algorithms
from Algorithm.Library.utils import plotter
import customtkinter as ctk
import base64
import requests
import pickle


def display_image(master, image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((530, 400))
        photo = ImageTk.PhotoImage(img)
        master.plot_label.configure(image=photo, text="")
        master.plot_label.image = photo
    except Exception as e:
        messagebox.showerror("Error", f"Could not display image: {e}")


def show_loading(master, message="Loading..."):
    if hasattr(master, "loading_frame"):
        return

    master.loading_frame = ctk.CTkFrame(master, fg_color="black", bg_color="black")
    master.loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    master.loading_label_overlay = ctk.CTkLabel(master.loading_frame, text=message)
    master.loading_label_overlay.configure(font=("Arial", 18, "bold"), text_color="white")
    master.loading_label_overlay.place(relx=0.5, rely=0.5, anchor="center")
    master.update_idletasks()
    
    
def hide_loading(master):
    if hasattr(master, "loading_frame"):
        master.loading_frame.destroy()
        delattr(master, "loading_frame")
        delattr(master, "loading_label_overlay")


def call_load(master):
    if not master.auth:
        messagebox.showerror(
            "Not connected",
            "Please connect to the database first."
        )
        return
    
    try:
        show_loading(master, "Running algorithm...")
        
        payload = {
            "auth": master.auth,
            "conn": master.db_conn
        }
        
        request = requests.post(f"{master.url}/load", json=payload)
        res = request.json()
        
        if res["status"] == "success":
            messagebox.showinfo("Success", res["message"])
            graph_bytes = base64.b64decode(res["data"])
            master.graph = pickle.load(io.BytesIO(graph_bytes))
        else:
            messagebox.showerror("Error", res["message"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to call operation: {e}")
    finally:
        hide_loading(master)
        algo = algorithms.Algorithms(master.graph)
        temp_dir = f"{os.getcwd()}/GUI/resources/"
        points = []
        points.append(master.graph.create_points(range(1, master.graph.nodes)))
        points.append(master.graph.center.coordinates)
        algo._plot_original(points, dir=temp_dir)
        temp_dir = f"{temp_dir}/Original.png"
        master.temp.append(temp_dir)
        display_image(master, temp_dir)
        

def call_divide(master):
    if not master.graph:
        messagebox.showerror(
            "Critical error",
            "There is no map to divide."
        )
        return
    
    capacity = master.capacity_entry.get()
    if not capacity:
        messagebox.showerror(
            "Critical error",
            "A truck capacity is needed."
        )
        return
    
    try:
        show_loading(master, "Running algorithm...")
        
        encoded_graph = pickle.dumps(master.graph)
        graph = base64.b64encode(encoded_graph).decode('utf-8')
        payload = {
            "graph": graph,
            "capacity": capacity
        }
        
        request = requests.post(f"{master.url}/divide", json=payload)
        res = request.json()
        
        if res["status"] == "success":
            messagebox.showinfo("Success", res["message"])
            master.zones = list(res["zones"])
        else:
            messagebox.showerror("Error", res["message"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to call operation: {e}")
    finally:
        hide_loading(master)


def call_solve(master):
    if not master.graph:
        messagebox.showerror(
            "Critical error",
            "There is no map to divide."
        )
        return
    
    try:
        show_loading(master, "Running algorithm...")
        
        graph = base64.b64encode(pickle.dumps(master.graph)).decode('utf-8')
        payload = {
            "graph": graph
        }
        
        request = requests.post(f"{master.url}/solve", json=payload)
        res = request.json()
        
        if res["status"] == "success":
            messagebox.showinfo("Success", res["message"])
            master.routes.append(list(res["route"]))
            master.values.append(float(res["value"]))
        else:
            messagebox.showerror("Error", res["message"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to call operation: {e}")
    finally:
        hide_loading(master)


def call_run(master):
    if not master.graph:
        messagebox.showerror(
            "Critical error",
            "There is no map to divide."
        )
        return
    
    capacity = master.capacity_entry.get()
    if not capacity:
        messagebox.showerror(
            "Critical error",
            "A truck capacity is needed."
        )
        return
    
    try:
        show_loading(master, "Running algorithm...")
        
        graph = base64.b64encode(pickle.dumps(master.graph)).decode('utf-8')
        payload = {
            "graph": graph,
            "capacity": capacity
        }
        
        request = requests.post(f"{master.url}/run", json=payload)
        res = request.json()
        
        if res["status"] == "success":
            messagebox.showinfo("Success", res["message"])
            master.routes = list(res["routes"])
            master.values = list(res["values"])
            map_bytes = base64.b64decode(res["map"])
            points = pickle.load(io.BytesIO(map_bytes))
            algo = algorithms.Algorithms(master.graph)
            algo.plot_multiple_paths(points,
                             dir=f"{os.getcwd()}/GUI/resources",
                             name="result")
        else:
            messagebox.showerror("Error", res["message"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to call operation: {e}")
    finally:
        hide_loading(master)


def call_send_routes(master):
    if not master.db_conn:
        messagebox.showerror(
            "Not connected",
            "Please connect to the database first."
        )
        return
    
    for i, r in enumerate(master.routes):
        try:
            show_loading(master, "Running algorithm...")
            
            payload = {
                "auth": master.auth,
                "db_conn": master.db_conn,
                "route": r,
                "tid": (i % master.t_num) + 1
            }
            
            request = requests.post(f"{master.url}/send-route", json=payload)
            res = request.json()
            
            if res["status"] == "error":
                messagebox.showerror("Error", res["message"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to call operation: {e}")
        finally:
            hide_loading(master)


class Operations:
    def __init__(self, master):
        master.title("Control Panel | Operations")

        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)
        master.grid_rowconfigure(8, weight=1)

        master.operations_label = ctk.CTkLabel(master, text="Available Operations")
        master.operations_label.configure(font=("Arial", 20, "bold"))
        master.operations_label.grid(row=0, column=1, columnspan=1, padx=20, pady=(20, 10), sticky="n")

        master.capacity_entry = ctk.CTkEntry(master, placeholder_text="Capacity")
        master.capacity_entry.grid(row=1, column=1, columnspan=1, padx=20, pady=10, sticky="ew")

        master.loading_label = ctk.CTkLabel(master, text="", text_color="gray")
        master.loading_label.grid(row=2, column=1, columnspan=1, padx=20, pady=5, sticky="w")

        master.load_btn = ctk.CTkButton(master, text="Load", command=lambda: call_load(master))
        master.load_btn.grid(row=3, column=1, padx=20, pady=5, sticky="ew")

        master.divide_btn = ctk.CTkButton(master, text="Divide", command=lambda: call_divide(master))
        master.divide_btn.grid(row=4, column=1, padx=20, pady=5, sticky="ew")

        master.solve_btn = ctk.CTkButton(master, text="Solve", command=lambda: call_solve(master))
        master.solve_btn.grid(row=5, column=1, padx=20, pady=5, sticky="ew")

        master.run_btn = ctk.CTkButton(master, text="Run", command=lambda: call_run(master))
        master.run_btn.grid(row=6, column=1, padx=20, pady=5, sticky="ew")

        master.send_route_btn = ctk.CTkButton(master, text="Send Routes", command=lambda: call_send_routes(master))
        master.send_route_btn.grid(row=7, column=1, padx=20, pady=5, sticky="ew")

        master.plot_label = ctk.CTkLabel(master, text="")
        master.plot_label.grid(row=0, column=2, rowspan=8, padx=(20, 20), pady=20, sticky="nsew")
