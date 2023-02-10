import json
import numpy as np

layout = np.zeros(shape=(18, 11), dtype=int)

if __name__ == "__main__":
    with open('problems.json') as p:
        d = json.load(p)

    print(d['data'][10])
    for k, v in d['data'][10].items():
        print(k, v)

    for hold in d['data'][10]['moves']:
        x, y = hold['description'][0], hold['description'][1:]
        x, y = ord(x) - 65, int(y) - 1
        layout[17 - y, x] = 1
    print(layout)
