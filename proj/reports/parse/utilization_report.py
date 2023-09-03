from pprint import pprint
from argparse import ArgumentParser
from common import save_update_report, add_filename_to_path_if_needed


def parse_utilization_report(utilization_report_filepath: str, result_filepath: str, design_name: str):
    utilization_report_filepath = add_filename_to_path_if_needed(
        utilization_report_filepath, "digilent_arty_utilization_place.rpt"
    )
    
    utilization = {}

    print(f"Searching utilization numbers in {utilization_report_filepath}")
    for line in open(utilization_report_filepath, "r"):
        line = line.strip()

        try:
            value = float(line.split("|")[-2].strip())
        except:
            continue

        if line.startswith("| Slice LUTs                 |"):
            utilization["LUT"] = value
        if line.startswith("| LUT as Memory                              |"):
            utilization["LUT_RAM"] = value
        if line.startswith("| Slice Registers            |"):
            utilization["FF"] = value
        if line.startswith("| Block RAM Tile    |"):
            utilization["BRAM"] = value
        if line.startswith("| DSPs           |"):
            utilization["DSP"] = value
    print(f"Parsed utilizations:")
    pprint(utilization)

    save_update_report(result_filepath, design_name, utilization)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--design_name", "-d", required=True)
    args = parser.parse_args()

    parse_utilization_report(args.input, args.output, args.design_name)
