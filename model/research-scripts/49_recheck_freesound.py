from pathlib import Path
from importlib import import_module
import json
m=import_module('26_curate_rail');m.ROOT=Path('experiments/us-freight-candidates/freesound-recheck-2026-11-10')
r=m.fetch((157864,('markedit','train_review','US Buda Texas; prior blocked source, one controlled recheck after four days')))
print(json.dumps(r,indent=2))
