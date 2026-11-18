# Dataset layout

The four benchmark datasets are **not** bundled with this package. Together they
are about 600 MB, and their providers' terms do not allow us to redistribute
them here. Download them yourself and place them under this folder (`data/`),
or point `main` at any other folder with the same layout:

```matlab
main('Salinas', 'D:/hsi', 'results', 1001:1010);   % cube/label files under D:/hsi
```

`data/` is ignored by git except for `data/splits/`, so a local copy of the
datasets never ends up in a commit.

## Files and checksums

The SHA-256 values below are the exact files used for the reported runs. They
let you confirm that your download and conversion produced the same arrays; a
resave with a different MAT version changes the hash even when the arrays match.

| Folder | File | MAT variable | Size | SHA-256 |
|---|---|---|---:|---|
| `01_Indian_Pines` | `Indian_pines_corrected.mat` | `indian_pines_corrected` | 5.7 MB | `ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939` |
| `01_Indian_Pines` | `Indian_pines_gt.mat` | `indian_pines_gt` | 1.1 KB | `65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c` |
| `02_Salinas` | `Salinas_corrected.mat` | `salinas_corrected` | 26 MB | `44e16023de8a858389b5653d8dcf9afe5a75d4b0c0ea2494fa757201607ef872` |
| `02_Salinas` | `Salinas_gt.mat` | `salinas_gt` | 4.3 KB | `ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2` |
| `WHU-Hi-HongHu` | `WHU_Hi_HongHu.mat` | `WHU_Hi_HongHu` | 404 MB | `394eebb36e70931711b02d127f73c641ad67a83a5bbfc5294b695b2f7336dfce` |
| `WHU-Hi-HongHu` | `WHU_Hi_HongHu_gt.mat` | `WHU_Hi_HongHu_gt` | 10 KB | `afb4b455e078e308b64a51e4e1a9552ddffd2f8519eaa311ded94d74a96c2a41` |
| `Houston` | `Houston.mat` | `Houston` | 159 MB | `a3642fdc17620988d675949b61cb49949dac76df27f2c73aa68acfce6a9c80d3` |
| `Houston` | `Houston_gt.mat` | `Houston_gt` | 11 KB | `d3d380ffc1332683c8baa7b359f097f7d158b278b9d4b8e73f34934bdc29f4f2` |

## Array conventions

- Cube: `H × W × B` numeric array, band-last.
- Labels: `H × W` nonnegative integers, `0` for unlabeled pixels, positive
  integers for classes. Non-contiguous class IDs are supported.
- Houston expects a single combined label map for random sampling. The
  provider's predefined train/test maps are not merged automatically.

## Where to obtain them

| Dataset | Provider |
|---|---|
| Indian Pines, Salinas | Computational Intelligence Group, University of the Basque Country |
| WHU-Hi-HongHu | RSIDEA group, Wuhan University |
| Houston 2013 | IEEE GRSS Data Fusion Contest (requires a request) |

## Splits

`data/splits/<Dataset>_splits.mat` holds the fixed train/test linear indices
shipped with this package: all four datasets, seeds 1001, 1003, 1005, 1007 and
1009. Each file records the relative cube path and the SHA-256 of the label
file it was built from, so a mismatched dataset is detected rather than silently
scored against the wrong partitions.
