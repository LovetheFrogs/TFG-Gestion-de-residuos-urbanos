import psycopg2
import os
from getpass import getpass

def main():
    print("WARNING!!! Running this script will overwrite current data")
    print("Make sure the data is placed on the /data folder")
    print('File must have the name data and extension ".dat"')
    print()
    connect()
    print("Closing...")
    
def connect():
    print("Input database data to upload")
    dbname = input("Database name: ")
    user = input("User: ")
    password = getpass()
    host = input("Hostname: ")
    port = input("Port: ")
    
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    
    cur = conn.cursor()
    print("Deleting data...")
    cur.execute("TRUNCATE TABLE EDGE, NODE RESTART IDENTITY CASCADE;")
    conn.commit()
    
    print("Executing querry...")
    upload(cur)
    
    conn.commit()
    cur.close()
    conn.close()
    
    
def upload(cur):
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
    

if __name__ == '__main__':
    main()
