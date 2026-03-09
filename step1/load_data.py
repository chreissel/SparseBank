from gwpy.timeseries import TimeSeries, TimeSeriesDict
import yaml
from utils import load_config
from pathlib import Path
import os
import subprocess
import requests
from gwpy.segments import Segment, SegmentList
from functools import reduce
import operator
from tqdm import tqdm

BASE_URL = "https://gwosc.org/api/v2/runs/O3a/timelines"

def fetch_segments(config, detector="H1"):
    timeline = f"{detector}_DATA"
    url = f"{BASE_URL}/{timeline}/segments"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    segments = []
    for seg in data["results"]:
        start = int(seg['start'])
        end = int(seg['stop'])
        duration = end - start
        segments.append(Segment(start, end))

    return SegmentList(segments)


def load_data(config, data_dir: str):

    background_dir = data_dir / "background_data"
    background_dir.mkdir(parents=True, exist_ok=True)

    segments = {}
    for ifo in config.general.ifos:
        segments[ifo] = fetch_segments(config, ifo)
    network_segments = reduce(operator.and_, segments.values())

    #segments = [
    #    (1240579783, 1240587612),
    #    (1240594562, 1240606748),
    #    (1240624412, 1240644412),
    #    (1240644412, 1240654372),
    #    (1240658942, 1240668052),
    #]

    for (start, end) in tqdm(network_segments):
        duration = end - start
        if duration >= config.general.waveform_duration:
            fname = background_dir / f"background-{start}-{duration}.hdf5"
            if fname.exists():
                continue

            ts_dict = TimeSeriesDict()
            for ifo in config.general.ifos:
                ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
            ts_dict = ts_dict.resample(config.general.sample_rate)
            ts_dict.write(fname, format="hdf5")

if __name__ == '__main__':
    config = load_config(config_path='configs/config_BNS.yaml')
    data_dir = Path("./data")
    load_data(config, data_dir)
