import linecache
import os
from tkinter import ttk
import customtkinter as ctk

from home import Home
from operations import Operations
from setting import Settings
from login import Login

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")
NUM_OF_WINDOWS = 4


def load_url():
    if not os.path.exists("resources//config.conf"):
        return "http://127.0.0.1:5000"

    data = linecache.getline("resources//config.conf", 3, module_globals=None)
    if data == "" or data == "None\n":
        return "http://127.0.0.1:5000"

    return str(data.strip())


def load_db_data():
    db_conn = {
        "user": "",
        "password": "",
        "host": "localhost",
        "dbname": "logistics",
        "port": "5432"
    }
    
    if not os.path.exists("resources//config.conf"):
        return db_conn

    data = linecache.getline("resources//config.conf", 2, module_globals=None)
    if data == "" or data == "None\n":
        return db_conn

    db_conn["host"] = linecache.getline("resources//config.conf", 4, module_globals=None).strip()
    db_conn["port"] = linecache.getline("resources//config.conf", 5, module_globals=None).strip()
    db_conn["dbname"] = linecache.getline("resources//config.conf", 6, module_globals=None).strip()

    return data


def load_data():
    if not os.path.exists("resources//config.conf"):
        return 15

    data = linecache.getline("resources//config.conf", 2, module_globals=None)
    if data == "" or data == "None\n":
        return 15

    return int(data.strip())


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window logic
        self.url = load_url()
        self.t_num = load_data()
        self.screens = [True, False, False, False]
        self.db_conn = load_db_data()
        self.auth = False
        self.generated_data = [True, False, False, False]
        self.routes = []
        self.temp = []
        self.values = []
        self.zones = None
        self.subgraphs = None
        self.graph = None

        theme = linecache.getline("resources//config.conf", 1, module_globals=None)
        if theme != "":
            ctk.set_appearance_mode(theme.strip())

        # Configure window
        self.title("Control Panel")
        self.geometry(f"{1100}x{500}")
        self.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.resizable(0, 0)

        self.style = ttk.Style(self)
        self.style.configure("Treeview.Heading", font=("Arial", 18, "bold"))
        self.style.configure("Treeview", font=("Arial", 16))
        ttk.Style()

        self.style.theme_use("default")

        self.style.configure("Treeview",
                             background="#2a2d2e",
                             foreground="white",
                             rowheight=25,
                             fieldbackground="#343638",
                             bordercolor="#343638",
                             borderwidth=0)
        self.style.map('Treeview', background=[('selected', '#22559b')])

        self.style.configure("Treeview.Heading",
                             background="#565b5e",
                             foreground="white",
                             relief="flat")
        self.style.map("Treeview.Heading",
                       background=[('active', '#3484F0')])

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure((0, 1), weight=1)

        # Creating sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.sidebar_logo = ctk.CTkLabel(self.sidebar_frame, text="Control Panel")
        self.sidebar_logo.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_home_button = ctk.CTkButton(self.sidebar_frame, command=self.home_event, text="Home")
        self.sidebar_home_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.sidebar_profile = ctk.CTkButton(self.sidebar_frame, command=self.login_event, text="Login")
        self.sidebar_profile.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.sidebar_problem = ctk.CTkButton(self.sidebar_frame, command=self.operations_event, text="Operations")
        self.sidebar_problem.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.sidebar_settings = ctk.CTkButton(self.sidebar_frame, command=self.settings_event, text="Settings")
        self.sidebar_settings.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        Home(self)

    def home_event(self):
        self.clear()
        Home(self)
        for i in range(NUM_OF_WINDOWS):
            self.screens[i] = False
        self.screens[0] = True

    def login_event(self):
        self.clear()
        Login(self)
        for i in range(NUM_OF_WINDOWS):
            self.screens[i] = False
        self.screens[1] = True

    def operations_event(self):
        self.clear()
        Operations(self)
        for i in range(NUM_OF_WINDOWS):
            self.screens[i] = False
        self.screens[2] = True

    def settings_event(self):
        if self.screens[3]:
            return
        else:
            self.clear()
            Settings(self)
            for i in range(NUM_OF_WINDOWS):
                self.screens[i] = False
            self.screens[3] = True

    def clear(self):
        widgets = self.winfo_children()

        for child in widgets:
            if child == self.sidebar_frame:
                continue
            else:
                child.destroy()

    def cleanup(self):
        with open(f'{os.getcwd()}/GUI/resources//config.conf', 'w') as f:
            f.writelines([str(ctk.get_appearance_mode()), "\n" + 
                          str(self.t_num), "\n" +
                          str(self.url), "\n" +
                          str(self.db_conn["host"]), "\n" +
                          str(self.db_conn["port"]), "\n" +
                          str(self.db_conn["dbname"])])

        for f in self.temp:
            os.remove(f)
        self.destroy()
