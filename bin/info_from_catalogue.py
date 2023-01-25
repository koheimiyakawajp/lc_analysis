#!/usr/bin/env python3

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astroquery.gaia import Gaia
import sys

from astroquery.vizier import Vizier
from astroquery.ipac.irsa import Irsa
from astropy.table import Table


def k2id_to_cood(k2id):
    filelist    = "./k2ticxmatch_20210831.csv"
    tarlist     = np.loadtxt(filelist, comments='tid', delimiter=",", usecols=[0,1,2,3], dtype='unicode')
    hittar      = tarlist[(tarlist[:,1]==str(int(k2id)))][0]
    
    ra          = hittar[2]
    dec         = hittar[3]

    return ra,dec


def get_gaia(k2id, rad=0.1):
    
    ra,dec  = k2id_to_cood(k2id)

    Gaia.MAIN_GAIA_TABLE="gaiadr3.gaia_source"
    Gaia.ROW_LIMIT  = 1
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    radius  = u.Quantity(rad, u.deg)

    j   = Gaia.cone_search_async(coord, radius)
    r   = j.get_results()
    
    plx     = r['parallax'][0]
    plx_er  = r['parallax_error'][0]
    bprp    = r['bp_rp'][0]
    ruwe    = r['ruwe'][0]

    return plx,plx_er,bprp,ruwe

def get_2mass(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    table = Irsa.query_region(coord, catalog="fp_psc", spatial="Cone",radius=rad*u.deg)
    #for i in table.columns:
    #    print(i)

    jmag    = table["j_m"][0]
    jmag_er = table["j_cmsig"][0]

    hmag    = table["h_m"][0]
    hmag_er = table["h_cmsig"][0]

    kmag    = table["k_m"][0]
    kmag_er = table["k_cmsig"][0]

    return jmag,jmag_er,hmag,hmag_er,kmag,kmag_er

def get_tycho(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    result = Vizier.query_region(coord, radius=rad*u.deg,
                                 catalog='I/239/tyc_main')
    vmag    = result[0]['Vmag'][0]

    return vmag

def get_lamost(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    result = Vizier.query_region(coord, radius=rad*u.deg,
                                 catalog='I/239/tyc_main')
    vmag    = result[0]['Vmag'][0]

    return vmag

if __name__=='__main__':
    k2id    = sys.argv[1]
    #Irsa.print_catalogs()
    #exit()
    #tables = Gaia.load_tables(only_names=True)
    #for table in (tables):
    #    print(table.get_qualified_name())
    #exit()
    #gaiadr3_table = Gaia.load_table('gaiadr3.gaia_source')
    #for column in gaiadr3_table.columns:
    #    print(column.name)
    #exit()

    #get_gaia(k2id, 0.01)
    #get_2mass(k2id, 0.01)
    get_tycho(k2id, 0.01)

