import argparse
import json
from itertools import product
from pathlib import Path

def generate_job_array(grid):
    keys = [key for key in grid.keys()]
    values = [grid[k1] for k1 in keys]
    output = []
    for v1 in product(*values):
        each = {}
        for k1,v1 in zip(keys,v1):
            each[k1] = v1
        output.append(each)
    return output

def pop(file):
    with open(file, 'r+') as f: # open file in read / write mode
        firstLine = f.readline() # read the first line and throw it out
        data = f.read() # read the rest
        f.seek(0) # set the cursor to the top of the file
        f.write(data) # write the data back
        f.truncate() # set the file size to the current size
    d = json.loads(firstLine)
    output = []
    for k,v in d.items():
        output.append(f"--{k}={v}")
    output = " ".join(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="ID to use")
    args = parser.parse_args()
    job_file = Path(f"{args.id}.job")
    if job_file.exists():
        fifo = pop(job_file)
        print(fifo)
    else:
        with open(f"{args.id}.grid") as f:
            grid = json.load(f)
        jobs = generate_job_array(grid)
        with open(job_file, "w") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")
        fifo = pop(job_file)
        print(fifo)
    return

if __name__ == "__main__":
    main()

