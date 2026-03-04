import re
from collections import defaultdict
import sys
import numpy as np

experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
current_experiment = None

for line in sys.stdin:
    line = line.strip()
    m = re.match('mpirun -n (\\d+) .\\/(.+?) (.+?) (.+?) (.+?) (.+?) (.+?) (.+?) (\\d+?) (\\d+?) (\\d+?) (\\d+)', line)

    if m:
        nprocs = m.group(1)
        executable = m.group(2)
        algorithm = m.group(3)
        distribution = (m.group(4), m.group(5), m.group(6))
        a_replication_factor = int(m.group(7))
        b_replication_factor = int(m.group(8))
        c_replication_factor = int(m.group(9))
        replication_factors = (a_replication_factor, b_replication_factor, c_replication_factor)
        shape = (int(m.group(10)), int(m.group(11)), int(m.group(12)))

        if a_replication_factor == b_replication_factor == c_replication_factor:
            current_experiment = (distribution,algorithm,nprocs,replication_factors,shape)
        else:
            current_experiment = None

    m = re.match(
        r'command=torchrun --nproc_per_node=(\d+) \./(.+?) --stationary-method (\S+) '
        r'--a-partition (\S+) --b-partition (\S+) --c-partition (\S+) '
        r'--a-replication (\d+) --b-replication (\d+) --c-replication (\d+) '
        r'--m (\d+) --n (\d+) --k (\d+)',
        line,
    )

    if m:
        nprocs = m.group(1)
        executable = m.group(2)
        algorithm = m.group(3)
        if algorithm == 'stationary_c':
            algorithm = 'sc'
        elif algorithm == 'stationary_b':
            algorithm = 'sb'
        distribution = (m.group(4), m.group(5), m.group(6))
        a_replication_factor = int(m.group(7))
        b_replication_factor = int(m.group(8))
        c_replication_factor = int(m.group(9))
        replication_factors = (a_replication_factor, b_replication_factor, c_replication_factor)
        shape = (int(m.group(10)), int(m.group(11)), int(m.group(12)))

        if a_replication_factor == b_replication_factor == c_replication_factor:
            current_experiment = (distribution,algorithm,nprocs,replication_factors,shape)
        else:
            current_experiment = None

    m = re.match('Min .+? s \\((.+) GFLOPs\\)', line)

    if m and current_experiment is not None:
        (distribution,algorithm,nprocs,replication_factors,shape) = current_experiment
        experiments[nprocs][distribution][algorithm][replication_factors][shape] = float(m.group(1))

    m = re.match('Max Distributed GFLOPs: (.+)', line)

    if m and current_experiment is not None:
        (distribution,algorithm,nprocs,replication_factors,shape) = current_experiment
        experiments[nprocs][distribution][algorithm][replication_factors][shape] = float(m.group(1))

    m = re.match('Median Distributed GFLOPs: (.+)', line)

    if m and current_experiment is not None:
        (distribution,algorithm,nprocs,replication_factors,shape) = current_experiment
        experiments[nprocs][distribution][algorithm][replication_factors][shape] = float(m.group(1))

# Initialize data structures to store both GFLOPs and batch sizes
data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'gflops': [], 'batch_sizes': []})))

# Collect all possible batch sizes from the data
all_batch_sizes = set()
for nprocs in experiments.keys():
    for distribution in experiments[nprocs].keys():
        for algorithm in experiments[nprocs][distribution].keys():
            for replication_factors in experiments[nprocs][distribution][algorithm].keys():
                for shape in experiments[nprocs][distribution][algorithm][replication_factors].keys():
                    all_batch_sizes.add(shape[0])  # First element of shape tuple is batch size

# Sort batch sizes for consistent ordering
batch_sizes = sorted(all_batch_sizes)

# Collect data for each configuration
for m in batch_sizes:
    for nprocs in experiments.keys():
        for distribution in experiments[nprocs].keys():
            for algorithm in experiments[nprocs][distribution].keys():
                for replication_factors in experiments[nprocs][distribution][algorithm].keys():
                    shape_key = (m,49152,12288)
                    if shape_key in experiments[nprocs][distribution][algorithm][replication_factors]:
                        perf = experiments[nprocs][distribution][algorithm][replication_factors][shape_key]
                        data[distribution][algorithm][replication_factors]['gflops'].append(perf)
                        data[distribution][algorithm][replication_factors]['batch_sizes'].append(m)

data_optimized = defaultdict(lambda: defaultdict(lambda: tuple))

for distribution in data.keys():
    for algorithm in data[distribution].keys():
        # Get all replication factors and their performance vectors
        replication_factors_tuples = sorted(data[distribution][algorithm].keys())
        performance_vectors = [data[distribution][algorithm][r]['gflops'] for r in replication_factors_tuples]
        batch_size_vectors = [data[distribution][algorithm][r]['batch_sizes'] for r in replication_factors_tuples]

        # Get unique batch sizes across all configurations for this distribution/algorithm
        common_batch_sizes = sorted(set().union(*[set(bs) for bs in batch_size_vectors]))

        # Initialize results for each batch size
        optimal_results = []

        # For each batch size, find the best performing configuration
        for bs in common_batch_sizes:
            max_perf = float('-inf')
            best_rf = None

            # Check each configuration
            for rf_idx, rf in enumerate(replication_factors_tuples):
                try:
                    # Find where this batch size appears in this configuration's data
                    bs_idx = batch_size_vectors[rf_idx].index(bs)
                    perf = performance_vectors[rf_idx][bs_idx]
                    if perf > max_perf:
                        max_perf = perf
                        best_rf = rf
                except ValueError:
                    # This configuration doesn't have this batch size
                    continue

            if best_rf is not None:
                optimal_results.append((max_perf, best_rf, bs))

        # Unzip the results
        if optimal_results:
            optimal_performance, optimal_replication_factors, optimal_batch_sizes = zip(*optimal_results)
            data_optimized[distribution][algorithm] = (optimal_performance, optimal_replication_factors, optimal_batch_sizes)

data_optimized_selected = {}

for distribution in data_optimized.keys():
    if (data_optimized[distribution]['sc'][0][-1] if data_optimized[distribution]['sc'][0] else 0) > \
       (data_optimized[distribution]['sb'][0][-1] if data_optimized[distribution]['sb'][0] else 0):
        data_optimized_selected[distribution] = ('sc', *data_optimized[distribution]['sc'])
    else:
        data_optimized_selected[distribution] = ('sb', *data_optimized[distribution]['sb'])

print(data_optimized_selected)

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

markers = [
    'o', 's', '^', 'v', 'D', '*', 'X', 'P',
    'd', '<', '>', 'H', 'h', '1', '2', '3', '4', '+', 'x',
]

ANN_COLOR = 'black'    # any Matplotlib-recognised colour string
ANN_ALPHA = 0.40       # 0 = fully transparent, 1 = fully opaque
ANN_SIZE  = 9          # pt

def annotate(ax, xs, ys, labels, dy=3,
             color=ANN_COLOR, alpha=ANN_ALPHA, size=ANN_SIZE):
    """Write one *label* above each (x,y) with the chosen colour/alpha."""
    for x, y, rf_tuple in zip(xs, ys, labels):
        # Convert replication factor tuple to string
        label_text = str(max(rf_tuple))
        ax.annotate(f'{label_text}',
                    xy=(x, y),                # anchor: data point
                    xytext=(0, dy),           # shift upward by *dy* pt
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=size,
                    color=color,
                    alpha=alpha)              # <- transparency

fig, ax = plt.subplots()

tflops_per_tile = 67
num_tiles = 8

pretty_dist = {
    ('row', 'row', 'row'): 'UA - Row',
    ('column', 'column', 'column'): 'UA - Column',
    ('block', 'block', 'block'): 'UA - Block',
    ('row', 'column', 'column'): 'UA - Inner Prod.',
    ('column', 'row', 'row'): 'UA - Outer Prod.',
    ('traditional', 'traditional', 'traditional'): 'UA - Traditional',
    # ('column', 'row', 'block'): 'UA - CRB',
}

pretty_alg = {'sc': 'S-C', 'sb': 'S-B'}

data_optimized_filtered = list(filter(lambda x: x[0] in pretty_dist, data_optimized_selected.items()))

# Sort so colours/markers stay stable run-to-run
for idx, (dist, (alg, perf_vec, rf_vec, bs_vec)) in enumerate(sorted(data_optimized_filtered)):
    xs = bs_vec  # Use actual batch sizes
    ys = 100 * (np.array(perf_vec) / (num_tiles*tflops_per_tile * 1000))
    alg_tag = pretty_alg.get(alg, alg)
    dist_tag = pretty_dist.get(dist, 'x'.join(dist))
    label = f'{dist_tag} ({alg_tag})'                    # e.g. 4×1×0 (sc)

    ax.semilogx(xs, ys,
                marker=markers[idx % len(markers)],
                markerfacecolor='white',
                color=f'C{idx}',
                label=label)
    annotate(ax, xs, ys, rf_vec)

additional_data = [([1024, 2048, 4096, 8192],
                    [319644.63940534665, 323399.3125891105, 344543.37055557646, 346337.1633676299],
                    "DT - Row"),
                   ([1024, 2048, 4096, 8192],
                    [345228.60742313333, 379727.12236284197, 382918.1979303226, 335123.53904613975],
                    "DT - Column"),
                    ([1024, 2048, 4096, 8192],
                     [88364.4, 117819.0, 141383.0, 157092.0],
                     "COSMA-NCCL")
                   ]


if additional_data is not None:
    for idx,datum in enumerate(additional_data):
        xs, ys, label = datum
        ys = 100 * (np.array(ys) / (num_tiles*tflops_per_tile * 1000))
        marker_idx = len(data_optimized_filtered) + idx
        ax.semilogx(xs, ys,
                    marker=markers[marker_idx % len(markers)],
                    markerfacecolor='white',
                    color=f'C{marker_idx}',
                    label=label)

# Cosmetics – tweak as you wish
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Percent of Peak', fontsize=12)
ax.set_title(f'{num_tiles}xH100, FP32 GEMM, MLP-1 H=12K', fontsize=14)

ax.minorticks_off()
# Use actual batch sizes for x-ticks
all_batch_sizes = sorted(set(bs for _, (_, _, _, bs_vec) in data_optimized_filtered for bs in bs_vec))
ax.set_xticks(all_batch_sizes)
ax.xaxis.set_major_formatter(FormatStrFormatter('%s'))

ax.set_ylim(0, 100)
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
plt.rcParams.update({'font.size': 11})
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('out.pdf')      # → open out.pdf to see the chart
print('Plot written to out.pdf')
