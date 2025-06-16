import customtkinter as ctk
from tkinter import messagebox, ttk


def login(master):
    user = master.user_entry.get()
    password = master.password_entry.get()
    try:
        master.db_conn["user"] = user
        master.db_conn["password"] = password
        master.auth = True
        master.clear()
        Login(master)
    except Exception as e:
        messagebox.showerror("Connection Error", f"Failed to connect:\n{e}")
        master.db_conn = None
        master.auth = False
        


def logout(master):
    master.db_conn["user"] = ""
    master.db_conn["password"] = ""
    master.auth = False
    master.clear()
    Login(master)


class Login:
    def __init__(self, master):
        master.title("Control Panel | Login")
        if master.db_conn["user"] == "":
            master.login_label = ctk.CTkLabel(master, text="Connect to Database")
            master.login_label.configure(font=("Arial", 20, "bold"))
            master.login_label.grid(row=0, column=1, columnspan=1, padx=20, pady=10, sticky="nw")
            master.user_entry = ctk.CTkEntry(master, placeholder_text="Username")
            master.user_entry.grid(row=0, column=2, columnspan=1, padx=20, pady=10, sticky="n")
            master.password_entry = ctk.CTkEntry(master, placeholder_text="Password", show="*")
            master.password_entry.grid(row=0, column=3, columnspan=1, padx=20, pady=10, sticky="ne")
            master.login_btn = ctk.CTkButton(master, text="Connect", command=lambda: login(master))
            master.login_btn.grid(row=0, column=4, columnspan=2, padx=20, pady=10, sticky="ne")
            master.bind('<Return>', lambda e: login(master))

        else:
            master.welcome_label = ctk.CTkLabel(master, text="Connected to Database")
            master.welcome_label.configure(font=("Arial", 20, "bold"))
            master.welcome_label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)
            master.logout = ctk.CTkButton(master, text="Disconnect", command=lambda: logout(master))
            master.logout.grid(row=1, column=0, columnspan=2, padx=20, pady=10)