import json
import sys
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def main(args):
    found_items = []
    index = json.load(open('index.json', 'r'))
    n_found_items = 0
    for q in args:
        found_items_q = []
        for item in index:
            for label in set(item[2]):
                score = similar(q, label)
                if score >= 0.5:
                    found_items_q.append(item)
                    n_found_items +=1
                    break

        found_items.append(found_items_q)

    print(f'{n_found_items} items found.')

    found_items = sum(found_items, [])

    for item in found_items:
        print(item[1])


if __name__ == '__main__':
    main(sys.argv[1:])