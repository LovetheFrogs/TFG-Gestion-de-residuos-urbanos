import psycopg2
import os
import pickle
from getpass import getpass

def main():
    conn = connect()
    upload(conn)
    print("Closing...")
    
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
    cur.execute("TRUNCATE TABLE EDGE, NODE RESTART IDENTITY CASCADE;")
    conn.commit()
    
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
    