import sys
import json
import os
import re

def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-path':
        path = args[1]
        if path[-1] != '/':
            path += '/'
        print('Path:', path)
        if '\\' in path:
            path.replace("\\", "/", -1)
    else: 
        print('Usage: python print_generation.py -path <cwd_path_to_results.json>')
        return None
    cwd = os.getcwd()
    with open(cwd + path + 'results.json', 'r') as f:
        recipes = json.load(f)
    for recipe in recipes:
        print('Recipe # 1')
        print(recipe)
        print()
    return None

if __name__ == '__main__':
    main()
