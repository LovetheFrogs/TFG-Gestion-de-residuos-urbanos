# Daily Estimated Weight Job

The job connects to a PostgreSQL database, calculates the average weight for each node from the `HISTORIC_DATA` table, and updates the `ESTIMATED_DATA` table accordingly.

The script also supports saving and reusing connection data using a file located in `/data/db.dat`.

## Needed Structure

```text
/
├── job.py               
├── data/                
│   └── db.dat           
└── README.md            
```

Make sure the `/data` directory exists before running the script.

## Requirements

- Python 3.8+
- PostgreSQL server running
- Python packages:
  - `psycopg2`
  - `pickle` (built-in)
  - `getpass` (built-in)

Install dependencies (if not already installed):

```bash
pip install psycopg2-binary
```

## Required Database Tables

The following tables must exist beforehand:

```sql
CREATE TABLE NODE (
    NODE_ID SERIAL PRIMARY KEY,
    CENTER BOOLEAN NOT NULL,
    COORDINATES POINT NOT NULL
);

CREATE TABLE EDGE (
    EDGE_ID VARCHAR PRIMARY KEY,
    SPEED FLOAT,
    ORIGIN INTEGER REFERENCES NODE(NODE_ID),
    DESTINATION INTEGER REFERENCES NODE(NODE_ID)
);

CREATE TABLE HISTORIC_DATA (
    ENTRY SERIAL PRIMARY KEY,
    NODE_ID INTEGER REFERENCES NODE(NODE_ID),
    WEIGHT FLOAT,
    DATE TIMESTAMP
);

CREATE TABLE ESTIMATED_DATA (
    NODE_ID INTEGER PRIMARY KEY REFERENCES NODE(NODE_ID),
    WEIGHT FLOAT
);
```

## Scheduling Execution Daily at 00:00

### Ubuntu Server

Use `cron` to schedule the script.

1. Open the crontab editor:

```bash
crontab -e
```

2. Add the following line to execute the script daily at midnight:

```bash
0 0 * * * /usr/bin/python3 /path/to/your/job.py >> /path/to/your/log.txt 2>&1
```

> Replace `/usr/bin/python3` with the output of `which python3`  
> Replace `/path/to/your/` with the full path to the job

3. Ensure the script has execution permission and the data directory exists:

```bash
chmod +x /path/to/your/job.py
mkdir -p /path/to/your/data
```

### Windows

Use Task Scheduler:

1. Open **Task Scheduler** and click **Create Basic Task**.
2. Name it.
3. Trigger: **Daily**, repeat every 1 day, start at **12:00 AM**
4. Action: **Start a program**
   - Program/script: `python`
   - Add arguments: `C:\path\to\your\job.py`
   - Start in: `C:\path\to\your\`
5. Finish and ensure it’s enabled.

**Important:**
- Make sure `python` is in the system PATH.
- Ensure the `data` folder exists: `mkdir C:\path\to\your\data`

## Notes

- The password is **not** saved — you'll always be asked for it unless you modify the script.
