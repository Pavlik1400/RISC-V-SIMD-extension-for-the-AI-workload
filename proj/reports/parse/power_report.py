import os
from pprint import pprint
from argparse import ArgumentParser
from common import save_update_report, add_filename_to_path_if_needed


def parse_power_reports(power_report_filepath: str, result_filepath: str, design_name: str):
    power_report_filepath = add_filename_to_path_if_needed(
        power_report_filepath, "digilent_arty_power.rpt"
    )
    print(f"Searching power consumption numbers in {power_report_filepath}")
    power = -1

    for line in open(power_report_filepath, "r"):
        line = line.strip()
        if "Total On-Chip Power (W)" not in line:
            continue

        power = float(line.split("|")[2].strip())

    save_update_report(result_filepath, design_name, power)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--design_name", "-d", required=True)
    args = parser.parse_args()

    parse_power_reports(args.input, args.output, args.design_name)
