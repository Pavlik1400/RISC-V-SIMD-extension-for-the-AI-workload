from argparse import ArgumentParser
from power_report import parse_power_reports
from utilization_report import parse_utilization_report
from cycles import parse_cycles
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reports_dir", "-r", required=True)
    parser.add_argument("--cycles_input", "-c", required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--design_name", "-d", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    parse_cycles(args.cycles_input, output_dir / "cycles_hardware.json", args.design_name)
    parse_utilization_report(
        args.reports_dir, output_dir / "resources_utilization.json", args.design_name
    )
    parse_power_reports(args.reports_dir, output_dir / "power_consumption.json", args.design_name)
