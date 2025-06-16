import customtkinter as ctk


def change_theme(master):
    ctk.set_appearance_mode(master.theme_menu.get())


def change_btn(master):
    ctk.set_default_color_theme(master.btn_menu.get())
    master.cleanup()
    master.__init__()
    master.settings_event()


def change_scale(master):
    ctk.set_widget_scaling(int(master.scale_menu.get().replace("%", "")) / 100)


def change_trucks(master):
    master.t_num = int(master.items_menu.get())


class Settings:
    def __init__(self, master):
        master.title("Control Panel | Settings")
        master.settings_label = ctk.CTkLabel(master, text="Change the app settings")
        master.settings_label.configure(font=("Arial", 20, "bold"))
        master.settings_label.grid(row=0, column=1, padx=20, pady=10, sticky="n")

        master.theme = ctk.CTkLabel(master, text="Select app theme")
        master.theme.configure(font=("Arial", 16, "bold"))
        master.theme.grid(row=1, column=1, padx=10, pady=10, sticky="sw")
        master.theme_menu = ctk.CTkOptionMenu(master, values=["Light", "Dark", "System"],
                                              command=lambda x: change_theme(master))
        master.theme_menu.grid(row=1, column=2, padx=10, pady=10, sticky="se")

        master.btn = ctk.CTkLabel(master, text="Select button color")
        master.btn.configure(font=("Arial", 16, "bold"))
        master.btn.grid(row=2, column=1, padx=10, pady=10, sticky="nw")
        master.btn_menu = ctk.CTkOptionMenu(master, values=["blue", "green", "dark-blue"],
                                            command=lambda x: change_btn(master))
        master.btn_menu.grid(row=2, column=2, padx=10, pady=10, sticky="ne")

        master.scale = ctk.CTkLabel(master, text="Select app scaling")
        master.scale.configure(font=("Arial", 16, "bold"))
        master.scale.grid(row=3, column=1, padx=10, pady=10, sticky="nw")
        master.scale_menu = ctk.CTkOptionMenu(master, values=["80%", "90%", "100%", "110%", "120%"],
                                              command=lambda x: change_scale(master))
        master.scale_menu.grid(row=3, column=2, padx=10, pady=10, sticky="ne")

        master.items = ctk.CTkLabel(master, text="Select number of trucks")
        master.items.configure(font=("Arial", 16, "bold"))
        master.items.grid(row=4, column=1, padx=10, pady=10, sticky="nw")
        master.items_menu = ctk.CTkOptionMenu(master, values=["10", "11", "12", "13", "14", "15", "16", "17", "18"],
                                              command=lambda x: change_trucks(master))
        master.items_menu.grid(row=4, column=2, padx=10, pady=10, sticky="ne")
