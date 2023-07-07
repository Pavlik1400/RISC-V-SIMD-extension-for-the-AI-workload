import ast
from pprint import pprint
from argparse import ArgumentParser
from common import save_update_report


def parse_cycles_old(cycles_filepath: str):
    # state = "looking_for_cycles"
    state = "looking_for_anchor"

    results = {}
    cur_cfu_name = None

    for line in open(cycles_filepath, "r"):
        line = line.strip()
        if line.startswith("// CFU V") and state == "looking_for_anchor":
            state = "looking_for_cycles"
            cur_cfu_name = line.strip("//").strip()
            results[cur_cfu_name] = None
        if line.startswith("[") and state == "looking_for_cycles":
            state = "looking_for_anchor"
            results[cur_cfu_name] = ast.literal_eval(line)

    pprint(results)


def parse_cycles(cycles_filepath: str, result_filepath: str, experiment_name: str):
    cycles = []

    print(f"Searching {cycles_filepath} for cycles prints")
    for line in open(cycles_filepath, "r"):
        line = line.strip()

        # Found line with cycles print
        if line.endswith("cycles total"):
            n_cycles = int(line.split("(")[1].split(")")[0].strip())
            cycles.append(n_cycles)
            print(".", end="")
    print()
    print(f"Cycles: {cycles}")

    save_update_report(result_filepath, experiment_name, cycles)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--experiment_name", "-e", required=True)
    args = parser.parse_args()

    parse_cycles(args.input, args.output, args.experiment_name)
