"""Acquire documented-location US freight candidates with frozen creator roles.

Uses the common provenance/acquisition audit. Does not change v2 labels, train
models, purchase audio, or inspect reserved audio with a classifier.
"""
from importlib import import_module
from pathlib import Path

def main():
    c=import_module('26_curate_rail')
    c.ROOT=Path('experiments/us-freight-candidates/acquired')
    c.CANDIDATES={
        580528:('the_toilet_guy','train_review','Jamestown, North Carolina, USA; lineside freight; binaural recording'),
        256259:('jstorm_57','train_review','Brady, Washington, USA; moderate-speed freight; creator reports crossing bell shuts off during pass'),
        667746:('felix.blume','train_review','Dekalb, Illinois, USA; very slow close freight; road traffic and occasional voices reported'),
        383465:('casemundy','train_review','Creator pack Sounds of Houston; overhead bridge perspective and distant freeway; city attribution should be checked before asserting location'),
        157864:('markedit','train_review','Buda, Texas, USA; edited approach/pass with horn and trailing wheel rumble; distinguish from raw sibling'),
        144090:('timgormly','reserved_review','Warrensburg, Missouri, USA; lineside cargo pass; reserve creator before acoustic screening'),
        547950:('greysound','reserved_review','Delevan, New York, USA; distant freight around 500 m away with intervening houses; reserve distance condition'),
        194493:('chripei','reserved_review','Indiana, USA; freight several miles away with horns; reserve distant condition'),
    }
    c.main()

if __name__=='__main__':main()
