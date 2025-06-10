import psycopg2
import os
import pickle
import random
from datetime import date, timedelta
from getpass import getpass

def main():
    option = print_menu()
    if (option == '4'):
        print("Closing...")
        exit()
    conn = connect()
    
    if (option == '1'):
        upload(conn)
    elif (option == '2'):
        load(conn)
    elif (option == '3'):
        job(conn)
    
    print("Run another option? (Y/n)")
    op = input("> ")
    if (op != "n"):
        main()
    else:
        print("Closing...")
    

def print_menu():
    option = ""
    
    while (option not in ["1", "2", "3", "4"]):
        print("WARNING!!! Running this script will overwrite current data")
        print("Make sure the data is placed on the /data folder")
        print('File must have the name data and extension ".dat"')
        print("------------------------------------------------------------")
        print("Select an option from the menu: ")
        print("1) Upload map data to the database")
        print("2) Load sample historic data")
        print("3) Run job from historic data")
        print("4) Exit")
        option = input("> ")
    
        if (option not in ["1", "2", "3", "4"]):
            print("Invalid option")
    
    return option
    

def connect():
    db_data = {
        "dbname": "", # type: ignore
        "host": "", # type: ignore
        "port": "", # type: ignore
        "user": "" # type: ignore
    }
    
    psw, option = "", ""
    
    if (os.path.exists(f"{os.getcwd()}/data/db.dat")):
        print("A file with database data has been found. Use it? (Y/n)")
        option = input("> ")
        if (option != "n"):
            with open(f"{os.getcwd()}/data/db.dat", "rb") as f:
                db_data = pickle.load(f)
    
    if (option == "n" or not os.path.exists(f"{os.getcwd()}/data/db.dat")):
        print("Input database data to upload")
        db_data["dbname"] = input("Database name: ")
        db_data["user"] = input("User: ")
        db_data["host"] = input("Hostname: ")
        db_data["port"] = input("Port: ")
        
    psw = getpass(prompt=f"Password for user {db_data['user']}: ")
    
    conn = psycopg2.connect(
        dbname=db_data["dbname"],
        user=db_data["user"],
        password=psw,
        host=db_data["host"],
        port=db_data["port"]
    )
    
    if (not os.path.exists(f"{os.getcwd()}/data/db.dat")):
        print("Save the database data? (Y/n)")
        option = input("> ")
        if (option != "n"):
            with open(f"{os.getcwd()}/data/db.dat", "wb") as f:
                pickle.dump(db_data, f)
    
    return conn


def upload(conn):
    cur = conn.cursor()
    
    print("Deleting data...")
    cur.execute("TRUNCATE TABLE EDGE, NODE RESTART IDENTITY CASCADE;")
    conn.commit()
    
    print("Executing querry...")
    
    with open(f"{os.getcwd()}/data/data.dat") as f:
        n = int(f.readline().strip())
        for _ in range(n):
            node_data = f.readline().strip().split()
            cur.execute("""
                INSERT INTO NODE (NODE_ID, CENTER, COORDINATES)
                VALUES (%s, %s, %s) RETURNING NODE_ID;
            """, (node_data[0], int(node_data[0])==0,
                  f"({float(node_data[1])}, {float(node_data[2])})"))
        m = int(f.readline().strip())
        for _ in range(m):
            edge_data = f.readline().strip().split()
            edge_id = f"{int(edge_data[1]):04}{int(edge_data[2]):04}"
            speed = edge_data[0]
            cur.execute("""
                INSERT INTO EDGE (EDGE_ID, SPEED, ORIGIN, DESTINATION) 
                VALUES (%s, %s, %s, %s)
            """, (edge_id, speed,
                  edge_data[1], edge_data[2]))
            
    conn.commit()
    cur.close()
    conn.close()


def load(conn):
    cur = conn.cursor()
    
    print("Deleting data...")
    cur.execute("TRUNCATE TABLE HISTORIC_DATA RESTART IDENTITY CASCADE;")
    conn.commit()
    
    print("Executing querry...")
    
    cur.execute("SELECT COUNT(*) FROM NODE;")
    num_nodes = cur.fetchone()[0]

    print("Number of days for the sample")
    days = input("> ")

    start_date = date.today() - timedelta(days=days)

    for day in range(100):
        current_date = start_date + timedelta(days=day)
        for node_id in range(0, num_nodes):
            weight = round(random.uniform(50.0, 150.0), 2)
            cur.execute("""
                INSERT INTO HISTORIC_DATA (NODE_ID, WEIGHT, DATE)
                VALUES (%s, %s, %s)
            """, (node_id, weight, current_date))
    
    conn.commit()
    cur.close()
    conn.close()

    
def job(conn):
    cur = conn.cursor()
    
    print("Deleting data...")
    cur.execute("TRUNCATE TABLE ESTIMATED_DATA RESTART IDENTITY;")
    conn.commit()
    
    print("Executing querry...")
    
    cur.execute("""
        SELECT NODE_ID, AVG(WEIGHT) AS ESTIMATED_WEIGHT
        FROM HISTORIC_DATA
        GROUP BY NODE_ID
    """)
    estimates = cur.fetchall()

    cur.execute("TRUNCATE TABLE ESTIMATED_DATA RESTART IDENTITY;")
    for node_id, estimated_weight in estimates:
        cur.execute("""
            INSERT INTO ESTIMATED_DATA (NODE_ID, WEIGHT)
            VALUES (%s, %s)
        """, (node_id, round(estimated_weight, 2)))
        
    conn.commit()
    cur.close()
    conn.close()
    

if __name__ == '__main__':
    main()
    