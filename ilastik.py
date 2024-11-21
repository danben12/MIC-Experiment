import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Path to the ilastik executable
ilastik_executable = r'C:\Users\Owner\ilastik-1.3.3post3\ilastik.exe'

# Dictionary mapping each chip to its project file and directory
chip_data = {
    'C1': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C1.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C1Large_HDF5'
    },
    'C2': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C2.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C2Large_HDF5'
    },
    'C3': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C3.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C3Large_HDF5'
    },
    'C4': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C4.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C4Large_HDF5'
    },
    'C5': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C5.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C5Large_HDF5'
    },
    'C6': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C6.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C6Large_HDF5'
    },
    'C7': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C7.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C7Large_HDF5'
    },
    'C8': {
        'project_file': r'K:\BSF_0762024_Amp MIC\ilastik\area based segmetation - C8.ilp',
        'directory': r'K:\BSF_0762024_Amp MIC\HDF5\C8Large_HDF5'
    }
}

# Get the number of CPUs available
num_threads = os.cpu_count()
print(f"Number of threads: {num_threads}")

# Set the environment variables for the number of threads and memory usage
os.environ['LAZYFLOW_THREADS'] = str(num_threads)
os.environ['LAZYFLOW_TOTAL_RAM_MB'] = '245760'  # Set total RAM to 240GB
os.environ['LAZYFLOW_DISK_CACHE_SIZE'] = '102400'  # Set disk cache size to 100GB
os.environ['LAZYFLOW_BLOCK_SHAPE'] = '1,1,10000,10000,2'  # Set a larger block shape

def run_ilastik(input_data, project_file):
    print(f"Processing file: {input_data} with project: {project_file}")
    command = [
        ilastik_executable,
        '--headless',
        '--project=' + project_file,
        '--raw_data=' + input_data,
        '--export_source=Simple Segmentation'
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, env=os.environ.copy(), stdout=devnull, stderr=devnull)

def process_in_batches(data_files, project_file, batch_size):
    for i in range(0, len(data_files), batch_size):
        batch = data_files[i:i + batch_size]
        batch_start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(lambda file: run_ilastik(file, project_file), batch)
        batch_end_time = time.time()
        print(f"Batch {i // batch_size + 1} took {batch_end_time - batch_start_time:.2f} seconds")

if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    batch_size = 3  # Adjust the batch size as needed
    for chip, data in chip_data.items():
        input_data_files = [os.path.join(data['directory'], f) for f in os.listdir(data['directory']) if f.endswith('.h5')]
        process_in_batches(input_data_files, data['project_file'], batch_size)
    end_time = time.time()  # Record the end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")