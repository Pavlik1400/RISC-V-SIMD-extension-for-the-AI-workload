from pathlib import Path
from tools.data import generate_data_simc_v1, DataSIM_1_Config
from argparse import ArgumentParser

modulations = [
    "16QAM",
    "64QAM",
    "8PSK",
    "B-FM",
    "BPSK",
    "CPFSK",
    "DSB-AM",
    "GFSK",
    "PAM4",
    "QPSK",
    "SSB-AM",
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, required=True, help="Where to save data")
    args = parser.parse_args()

    gen_config = DataSIM_1_Config(
        frames_per_mod_type=10_000,
        SNR=10.0,
        fs=200e3,
        sps=8.0,
        spf=1024,
        trans_delay=50,
        rician_path_delays=[0.0 / 200e3, 1.8 / 200e3, 3.4 / 200e3],
        rician_averate_path_gains=[0.0, -2.0, -10.0],
        rician_maximum_clockoffset=5.0,
        rician_k_factor=4.0,
        rician_maximum_doppler_shift=4.0,
    )
    generate_data_simc_v1(gen_config, modulations, Path(args.save_path))