from argparse import ArgumentParser
from pprint import pprint
from typing import Dict


CYCLES_PER_TICK = 1024


def to_latex_table_input(cycles: Dict, cycles_perc: Dict):
    assert sorted(list(cycles.keys())) == sorted(list(cycles_perc.keys()))
    result = ""
    for layer_name in cycles:
        cur_cycles = cycles[layer_name]
        cur_cycles_perc = cycles_perc[layer_name]
        result += f" {layer_name} & {cur_cycles} & {cur_cycles_perc} \\\\  [1ex]\n"
    return result


def main(input_path: str):
    layers_cycles = {}
    for i, line in enumerate(open(input_path)):
        if i == 0:  # Skip header
            continue

        if line.count(',') != 2:
            continue
        _, name, ticks = line.strip().split(",")
        cur_cycles = layers_cycles.get(name, 0)
        layers_cycles[name] = cur_cycles + int(ticks) * CYCLES_PER_TICK
    print("Per layer cycles:")
    pprint(layers_cycles)

    perc_cycles = {}
    all_ticks = sum(layers_cycles.values())
    for name, ticks in layers_cycles.items():
        perc_cycles[name] = ticks / all_ticks * 100

    print("Per layer cycles percentage:")
    pprint(perc_cycles)

    print(to_latex_table_input(layers_cycles, perc_cycles))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    main(parser.parse_args().input)
