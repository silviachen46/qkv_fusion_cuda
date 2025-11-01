import argparse, pandas as pd, matplotlib.pyplot as plt, os
def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index=['batch','new_tokens'], columns='fusion', values='tokens_per_sec', aggfunc='mean')
    pivot.plot(kind='bar', rot=0)
    plt.title('Tokens/s: fusion on vs off')
    plt.ylabel('tokens/s')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tokens_per_sec.png'))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/bench.csv')
    ap.add_argument('--out', default='results/plots')
    args = ap.parse_args()
    main(args.csv, args.out)
