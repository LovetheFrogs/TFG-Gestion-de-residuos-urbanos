import customtkinter as ctk


class Home:
    def __init__(self, master):
        master.title("Control Panel | Home")

        master.home_label = ctk.CTkLabel(master, text="Welcome to the control panel")
        master.home_label.configure(font=("Arial", 20, "bold"))
        master.home_label.grid(row=0, column=1, padx=20, pady=10, sticky="n")
        master.home_text = ctk.CTkLabel(master, text="To use the control panel, please login to the database"
                                                     "to start sending requests", wraplength=800)
        master.home_text.configure(font=("Arial", 14))
        master.home_text.grid(row=1, column=1, padx=20, pady=20, sticky="n")
