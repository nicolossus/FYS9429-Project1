#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

infiles_pattern = "hh_feature_data_*.csv"
outfile = "hh_sim_data.csv"

base_path = Path(__file__).resolve().parent
infiles_dir = base_path / "data"

files = infiles_dir.glob(infiles_pattern)

dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)
df.to_csv(base_path / "data" / outfile, index=False)
