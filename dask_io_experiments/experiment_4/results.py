import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(description="Outputs graph from experiment 4 csv file results. Remove header from file before processing.")
    parser.add_argument('results_filepath', 
        action='store', 
        type=str, 
        help='Path to csv file containing results')
    parser.add_argument('output_dir', 
        action='store', 
        type=str, 
        help='Path to output directory to store graph as image')
    parser.add_argument('--hdd-name', 
        action='store', 
        type=str, 
        dest='hdd_name',
        help='Name for hdd hardware. Default is Lustre.',
        default="Lustre")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    cols = ["hardware", "chunk shape", "split time(s)", "merge time(s)"]
    input_filename = args.results_filepath.split('/')[-1].split('.')[0]
    csv_data = pd.read_csv(args.results_filepath, header=None, names=cols)

    hdd_data = csv_data[csv_data["hardware"]=="HDD"].drop(columns=["hardware"])
    ssd_data = csv_data[csv_data["hardware"]=="SSD"].drop(columns=["hardware"])

    hdd_data_mean = hdd_data.groupby(["chunk shape"]).mean()
    hdd_data_std = hdd_data.groupby(["chunk shape"]).std()
    ssd_data_mean = ssd_data.groupby(["chunk shape"]).mean()
    ssd_data_std = ssd_data.groupby(["chunk shape"]).std()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharey=True)
    hdd_data_mean.plot(kind="bar", ax=axes[0], yerr=hdd_data_std, rot=0, title=args.hdd_name)
    ssd_data_mean.plot(kind="bar", ax=axes[1], yerr=ssd_data_std, rot=0, title="SSD")
    axes[0].grid(axis="y")
    axes[1].grid(axis="y")
    fig.suptitle('Results of experiment 1 comparing dask behavior when splitting/merging 3D arrays', fontsize=16)

    fig.savefig(os.path.join(args.output_dir, input_filename + ".png"))