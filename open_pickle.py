import pickle
from pathlib import Path
import os
import re
from datetime import datetime, timedelta
folder = Path('Grid_Search_1_logs')


differences = []
for file in folder.iterdir(): # remove this line after testing
    with open(file, 'rb') as file:

        first_line = file.readline()
        _ = file.readline()
        _ = file.readline()
        fourth_line = file.readline()

    line = first_line.decode("utf-8", errors="ignore")
    fourth_line = fourth_line.decode("utf-8", errors="ignore")
    match = re.search(r"\((.*?)\)", line)
    match_4 = re.search(r"\((.*?)\)", fourth_line)
    timestamp_4 = match_4.group(1)
    timestamp = match.group(1)
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
    dt_4 = datetime.strptime(timestamp_4, "%Y-%m-%d %H:%M:%S,%f")

    differences.append(dt_4-dt)

print(differences)
print(f'Average time difference: {sum(differences, timedelta()) / len(differences)}')