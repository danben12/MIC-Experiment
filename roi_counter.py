import time
import zipfile
import pandas as pd
import roifile
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count

def read_roi_metadata_from_zip(zip_file_path):
    roi_metadata = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.roi'):
                with zip_ref.open(file_name) as roi_file:
                    roi_data = roi_file.read()
                    roi = roifile.ImagejRoi.frombytes(roi_data)
                    coordinates = roi.coordinates()
                    metadata = {
                        'name': file_name,
                        'coordinates': coordinates
                    }
                    if roi.roitype == 8 or roi.roitype == 7:
                        area = calculate_polygon_area(coordinates)
                        metadata['area'] = area
                        if area > 4:
                            roi_metadata.append(metadata)
                        else:
                            continue
    return roi_metadata

def calculate_polygon_area(coordinates):
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    n = len(coordinates)
    area = 0.5 * abs(sum(x[i] * y[(i+1) % n] - y[i] * x[(i+1) % n] for i in range(n)))
    return area

def roi_to_polygon(roi_metadata):
    return Polygon(roi_metadata['coordinates'])

def check_bacteria_in_droplet(args):
    bacteria_id, bacteria_polygon, bacteria_area, droplets_polygons = args
    droplet_areas = {i: 0 for i in droplets_polygons}
    for droplet_id, droplet_polygon in droplets_polygons.items():
        if droplet_polygon.contains(bacteria_polygon):
            droplet_areas[droplet_id] += bacteria_area
            break
    return droplet_areas

def count_bacteria_in_droplets(bacteria_metadata, droplets_metadata):
    bacteria_polygons = {i: (roi_to_polygon(bacteria), bacteria['area']) for i, bacteria in enumerate(bacteria_metadata)}
    droplets_polygons = {droplet['name']: roi_to_polygon(droplet) for droplet in droplets_metadata}
    with Pool(cpu_count()) as pool:
        results = pool.map(check_bacteria_in_droplet,
                           [(bacteria_id, bacteria_polygon, bacteria_area, droplets_polygons) for bacteria_id, (bacteria_polygon, bacteria_area) in
                            bacteria_polygons.items()])
    droplet_areas = {i: 0 for i in droplets_polygons}
    for result in results:
        for droplet_id, area in result.items():
            droplet_areas[droplet_id] += area
    return droplet_areas

if __name__ == '__main__':
    base_path = r'K:\21012025_BSF obj x10'
    chips = [f'C{i}' for i in range(1, 9)]
    chips_full_names={'C1':"C1 - Control GFP T=",'C2':"C2 - 30 ug per ml GFP T=",'C3':"C3 - 10 ug per ml GFP T=",'C4':"C4 - 3.3 ug per ml GFP T=",'C5':"C5 - Control GFP T=",'C6':"C6 - 3.3 ug per ml GFP T=",'C7':"C7 - 30 ug per ml GFP T=",'C8':"C8 - 10 ug per ml GFP T="}
    hours = range(25)


    for chip in chips:
        t=time.time()
        results = []
        for hour in hours:
            t1=time.time()
            bacteria_path = f'{base_path}\\{chip}\\GFP\\bacteria validation\\{chips_full_names[chip]}{hour}.nd2 best z_RGB_Simple Segmentation.tif.zip'
            droplets_path = f'{base_path}\\{chip}\\Alexa T=0\\best LUT\\zip\\{chip}.zip'
            try:
                bacteria_roi = read_roi_metadata_from_zip(bacteria_path)
                droplets_roi = read_roi_metadata_from_zip(droplets_path)
                bacteria_in_droplets = count_bacteria_in_droplets(bacteria_roi, droplets_roi)
                for droplet_name, count in bacteria_in_droplets.items():
                    results.append({'Chip': chip, 'Hour': hour, 'Droplet': droplet_name, 'Bacteria Count': count})
                print(f'Processing {bacteria_path} and {droplets_path} {time.time() - t1:.2f} seconds')
            except FileNotFoundError as e:
                print(f'File not found: {e}')
                continue
        df = pd.DataFrame(results)
        df.to_csv(f'bacteria_counts{chip}.csv', index=False)
        print(f'{chip} took {time.time()-t:.2f} seconds')
