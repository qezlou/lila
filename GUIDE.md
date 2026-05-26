# LILA User Guide

**L**ine **I**ntensity map × **L**y-**A**lpha forest — forecasting 3D auto and cross power spectra for CO/CII line intensity maps, Ly-alpha forest tomography, and galaxy redshift surveys, using ASTRID (or TNG/MDPL2) hydrodynamic simulations.

---

## Installation

Requires Python < 3.9.

```bash
git clone https://github.com/qezlou/lila.git
cd lila
pip install -e .
```

---

## 1. Downloading the ASTRID Subhalo Catalog

The package reads SubFind group catalogs in HDF5 format. For the ASTRID simulation, these are available through the PSC portal.

### Steps

1. Go to [https://astrid-portal.psc.edu/simulation/1/](https://astrid-portal.psc.edu/simulation/1/)
2. Navigate to the snapshot of interest (e.g. z = 2.5).
3. Download the **SubFind** group catalog files. These are the `fof_subhalo_tab_*.hdf5` files (one or more chunks per snapshot).
4. Place all chunks for a snapshot in a directory named `groups_<snap>/`, where `<snap>` matches the snapshot identifier you will pass to the code (e.g. `z2.5`).

Your directory layout should look like:

```
<basepath>/
└── groups_z2.5/
    ├── fof_subhalo_tab_z2.5.0.hdf5
    ├── fof_subhalo_tab_z2.5.1.hdf5
    └── ...
```

### Required catalog fields

The code reads these fields from the `Subhalo` dataset inside each HDF5 file:

| Field | Description | Units (raw) |
|---|---|---|
| `SubhaloPos` | 3D position | ckpc/h (converted to cMpc/h internally) |
| `SubhaloMass` | Subhalo mass | 10^10 M_sun/h |
| `SubhaloVel` | Peculiar velocity | km/s |
| `SubhaloSFR` | Instantaneous SFR | M_sun/yr |
| `Header.Redshift` | Snapshot redshift | (HDF5 header attribute) |

The `sim_type='ASTRID'` flag in the code applies the `SubhaloPos / 1e3` unit conversion automatically.

---

## 2. Making a Line Intensity Map (CO, CII, or other lines)

The map-making classes all inherit from `lila.lim.MockLim`. The workflow is:
1. Point the class at your catalog directory.
2. Choose an emission model.
3. Instantiate the class — it builds the brightness temperature voxel map automatically.

### 2a. CO intensity map (COMAP survey)

Use `lila.comap.MockComap`. Two emission models are supported:

**COMAP+21 model** (recommended, data-driven, uses halo mass directly):

```python
from lila import comap

survey_params = {
    'beam_fwhm': 4.5,        # arcmin
    'freq_res': 31.25,        # MHz
    'nu_rest': 115.27,        # GHz, CO J=1-0
    'nu_co_rest': 115.27,
    'tempsys': 44.0,          # K
    'nfeeds': 19,
    'deltanu': 15.625,        # MHz
    'patch': 4,               # deg^2
    'tobs': 1500,             # hours
    'noise_per_voxel': 17.8,  # μK (COMAP Y5 projection)
}

COMAP21_params = {
    'A': -2.85,
    'B': -0.42,
    'C': 10**10.63,
    'M': 10**12.3,   # M_sun/h
    'sigma_co': 0.42,
}

co_mock = comap.MockComap(
    snap='z2.5',
    basepath='/path/to/astrid/subfind/',
    boxsize=[250, 250, 250],   # cMpc/h
    fine_Nmesh=[205, 205, 205],
    axis=3,                    # line-of-sight along z
    halo_type='Subhalo',
    sim_type='ASTRID',
    co_model='COMAP+21',
    COMAP21_params=COMAP21_params,
    survey_params=survey_params,
    mass_cut=10,               # log10(M_min / [M_sun/h])
    rsd=True,
    seed=42,
)

# The voxel map is now in co_mock.lim_map (an ArrayMesh in μK)
```

**Li+16 model** (SFR-based, requires the Behroozi+13 average SFR table):

```python
Li16_params = {
    'alpha': 1.17,
    'beta': -0.21,
    'sigma_co': 0.37,
    'delta_mf': 1,
    'sfr_type': 'behroozi+13',
    'behroozi_avsfr': '/path/to/behroozi+13/sfr_release.dat',
}

co_mock = comap.MockComap(
    ...,
    co_model='Li+16',
    Li16_params=Li16_params,
)
```

Alternative `sfr_type` values (use instantaneous SFR from the catalog instead of Behroozi):
- `'SubhaloSFR'` — uses the SubFind instantaneous SFR per subhalo
- `'GroupSFR'` — uses the group-level SFR

### 2b. CII intensity map (EXCLAIM survey)

Use `lila.exclaim.MockExclaim` with the Padmanabhan+19 model:

```python
from lila import exclaim

survey_params_cii = {
    'beam_fwhm': 4.33,        # arcmin
    'spec_res': 512,           # spectral resolution R = λ/Δλ
    'deltanu': 15.625,         # MHz
    'patch': 2.5,              # deg^2
    'tobs': 10.5,              # hours
    'noise_per_voxel': None,   # computed from instrument params
    'nu_rest': 1.901e6,        # MHz, CII 158 μm line
    'nu_co_rest': 115.27,
}

padmanabhan19_params = {
    'M1': 2.39e-5,
    'N1': 4.19e11,
}

cii_mock = exclaim.MockExclaim(
    snap='z2.5',
    basepath='/path/to/astrid/subfind/',
    boxsize=[250, 250, 250],
    fine_Nmesh=[205, 205, 205],
    axis=3,
    sim_type='ASTRID',
    cii_model='Padmanabhan+19',
    padmanabhan19_params=padmanabhan19_params,
    survey_params=survey_params_cii,
    rsd=True,
    seed=42,
)

# The voxel map is in cii_mock.lim_map (ArrayMesh in KJy sr^-1)
```

### 2c. Adding other emission lines

Subclass `lila.lim.MockLim` and implement three methods:

```python
from lila import lim

class MockMyLine(lim.MockLim):
    def get_res_par(self):
        # Return spatial resolution along LOS in cMpc/h
        ...

    def get_halo_luminosity(self):
        # Return array of luminosities in L_sun, one per halo
        ...

    def get_lim_map(self):
        # Build self.lim_map (ArrayMesh)
        lum_mesh = self.get_voxel_luminosity()   # CIC painting
        # apply your L → T_b conversion
        self.lim_map = ...
```

---

## 3. Ly-Alpha Forest Maps

The Ly-alpha forest maps are **not publicly available** and must be requested from the author (Mahdi Qezlou, `sumqezlou@gmail.com`). Two HDF5 files are needed per snapshot:

| File | Description | Key datasets |
|---|---|---|
| Noiseless map file | 3D δ_F field on a uniform grid | `map` (3D array), `redshift` (scalar) |
| Spectra file | Raw spectra used to build the map | `tau/H/1/1215`, `colden/H/1`, `Header` attrs (`hubble`, `omegam`, `omegab`) |

Once you have the files, load them with `lila.mock_lya.MockLya`:

```python
from lila import mock_lya

lya_mock = mock_lya.MockLya(
    noiseless_file='/path/to/map_ASTRID_true_0.25_z2.5.hdf5',
    spec_file='/path/to/spectra_ASTRID_noiseless_z2.5_1000_voxels.hdf5',
    boxsize=[250, 250, 250],   # cMpc/h
    dperp=2.5,                 # mean transverse sightline separation in cMpc/h
                               # (e.g. LATIS=2.5, PFS=3.7, eBOSS=13, DESI=10)
    sn=2,                      # average S/N per Angstrom
    transpose=(1, 0, 2),       # axis reordering if needed to match CO box orientation
    silent_mode=False,
)

# Noiseless δ_F map is in lya_mock.noiseless_map (numpy array)
```

HCD (Damped Lyman-alpha) masking is optional:

```python
HCD_mask = {'type': 'NHI', 'thresh': 10**19.7, 'vel_width': 200}  # mask log N_HI > 19.7
# or
HCD_mask = {'type': None}  # no masking
```

---

## 4. Power Spectra

`lila.stats.Stats` computes spherically averaged P(k), P(k,μ), and their cross-correlations. It takes any combination of `MockComap`, `MockLya`, and `MockGalaxy` instances.

### 4a. LIM auto power spectrum

```python
from lila import stats

st = stats.Stats(
    mock_lim=co_mock,
    kmin=0.01,       # h/cMpc
    kmax=1.0,
    dk=0.03,
    Nmu=30,
    los=[0, 0, 1],
)

pk_result = st.get_lim_pk(mode='1d')  # spherically averaged
print(pk_result.power['k'])           # k bins
print(pk_result.power['power'])       # P_CO(k) in μK^2 (cMpc/h)^3

# For 2D P(k, μ):
pk2d = st.get_lim_pk(mode='2d')
```

### 4b. Galaxy auto power spectrum

First build a galaxy overdensity map from the same catalog:

```python
from lila import mock_galaxy

gal_mock = mock_galaxy.MockGalaxy(
    snap='z2.5',
    basepath='/path/to/astrid/subfind/',
    boxsize=[250, 250, 250],
    axis=3,
    sim_type='ASTRID',
    halo_type='Subhalo',
    mass_cut=11.9,       # log10(M_min / [M_sun/h])
    Rz=0.007,            # redshift accuracy σ_z / (1+z)
    rsd=True,
    seed=42,
    silent_mode=False,
    save_path='gal_map_z2.5.hdf5',   # optional: cache the map to disk
)

st = stats.Stats(mock_galaxy=gal_mock, kmin=0.01, kmax=1.0, dk=0.03)
pk_gal = st.get_gal_pk(mode='1d')
```

Load a previously saved galaxy map to skip the halo-painting step:

```python
gal_mock = mock_galaxy.MockGalaxy(
    snap='z2.5',
    boxsize=[250, 250, 250],
    map_path='gal_map_z2.5.hdf5',
    Rz=0.007,
)
```

### 4c. CO × galaxy cross power spectrum

```python
st = stats.Stats(
    mock_lim=co_mock,
    mock_galaxy=gal_mock,
    kmin=0.01, kmax=1.0, dk=0.03, Nmu=30,
    los=[0, 0, 1],
    vol_ratio=1.0,   # set < 1 if simulated volume > survey volume
)

pk_cross = st.get_lim_gal_pk(mode='1d')
```

### 4d. CO × Ly-alpha forest cross power spectrum

```python
st = stats.Stats(
    mock_lim=co_mock,
    mock_lya=lya_mock,
    kmin=0.01, kmax=1.0, dk=0.03, Nmu=30,
)

pk_cross_lya = st.get_lim_lya_pk(mode='1d')
```

### 4e. Ly-alpha auto power spectrum

```python
st = stats.Stats(mock_lya=lya_mock, kmin=0.01, kmax=1.0, dk=0.03)
pk_lya = st.get_lya_pk(mode='1d')
```

### 4f. Signal-to-noise and uncertainties

```python
# Compute S/N for all signals at once
st = stats.Stats(mock_lim=co_mock, mock_galaxy=gal_mock, mock_lya=lya_mock,
                 kmin=0.01, kmax=1.0, dk=0.03, Nmu=30,
                 k_par_min=0.03)  # foreground wedge cut in h/cMpc

st.get_lim_sn()        # S/N for CO auto
st.get_lim_gal_sn()    # S/N for CO × galaxy
st.get_lim_lya_sn()    # S/N for CO × Lya
```

### 4g. Saving and loading results

```python
st.save_stat('results_z2.5.hdf5')

# Later, reload without rerunning the maps:
st2 = stats.Stats(z=2.5)
st2.load_stat('results_z2.5.hdf5')
```

The output HDF5 file stores `lim_pk/`, `gal_pk/`, `lya_pk/`, `lim_gal_pk/`, `lim_lya_pk/`, their noise estimates, uncertainties, and the git commit hash of the code version used.

---

## 5. Reference: Code and Notebooks

### Source modules

| File | Purpose |
|---|---|
| [src/lila/lim.py](src/lila/lim.py) | Base class `MockLim`: catalog loading, CIC painting, RSD |
| [src/lila/comap.py](src/lila/comap.py) | `MockComap`: CO emission (COMAP+21 and Li+16 models), L → T_b conversion |
| [src/lila/exclaim.py](src/lila/exclaim.py) | `MockExclaim`: CII emission (Padmanabhan+19 model) |
| [src/lila/mock_lya.py](src/lila/mock_lya.py) | `MockLya`: Ly-alpha forest δ_F maps, noise model, HCD masking |
| [src/lila/mock_galaxy.py](src/lila/mock_galaxy.py) | `MockGalaxy`: galaxy overdensity maps with redshift smearing |
| [src/lila/stats.py](src/lila/stats.py) | `Stats`: FFT auto/cross P(k), S/N, uncertainty estimates, save/load |
| [src/lila/inference.py](src/lila/inference.py) | Parameter inference on the biased linear power spectrum |
| [src/lila/plot.py](src/lila/plot.py) | Plotting utilities |

### Helper scripts

| File | Purpose |
|---|---|
| [helper_scripts/get_gal.py](helper_scripts/get_gal.py) | End-to-end script: CO + galaxy maps → P_CO, P_gal, P_{CO×gal}, S/N |
| [helper_scripts/get_lya.py](helper_scripts/get_lya.py) | End-to-end script: CO + Lya maps → P_CO, P_Lya, P_{CO×Lya}, S/N |
| [helper_scripts/get_latis_source_pk.py](helper_scripts/get_latis_source_pk.py) | Projected 2D source power spectrum for LATIS (requires non-public LATIS data) |

### Notebooks

| File | Contents |
|---|---|
| [notebooks/SN_results.ipynb](notebooks/SN_results.ipynb) | Forecast S/N ratios for CO auto, CO × Lya, CO × galaxies |
| [notebooks/Inference.ipynb](notebooks/Inference.ipynb) | Parameter inference on biased linear power spectrum |
| [notebooks/galaxy_selection.ipynb](notebooks/galaxy_selection.ipynb) | Analysis of HSC/CLAUDS photometry: redshift uncertainties and mass completeness via abundance matching |

### Key references

- CO emission model (COMAP+21): Chung et al. 2021, [arXiv:2111.05931](https://arxiv.org/abs/2111.05931)
- CO emission model (Li+16): Li et al. 2016, [arXiv:1503.08833](https://arxiv.org/abs/1503.08833)
- CII emission model: Padmanabhan 2019; Pullen et al. 2022
- Ly-alpha noise model: McQuinn & White 2011, [arXiv:1102.1752](https://arxiv.org/abs/1102.1752)
- Behroozi+13 SFR table: Behroozi et al. 2013
- ASTRID simulation: [astrid-portal.psc.edu](https://astrid-portal.psc.edu)
