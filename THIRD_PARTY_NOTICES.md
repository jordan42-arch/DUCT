# Third-Party Notices

The following notices accompany local baseline adaptations. They do not imply
endorsement by the upstream authors or grant rights to datasets.

| Upstream source | Local component | License notice |
| :--- | :--- | :--- |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | TimeMixer forecasting adaptation and TSLib-derived baseline components | [MIT, THUML](third_party/licenses/Time-Series-Library.txt) |
| [WPMixer](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer) | `models/WPMixer.py`, `layers/WPMixerDWT.py` | [MIT, SIS Lab](third_party/licenses/WPMixer.txt) |
| [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) | DWT routines in `layers/WPMixerDWT.py` | [MIT, Fergal Cotter](third_party/licenses/pytorch_wavelets.txt) |

Original source attribution comments have been retained. This release uses
discrete wavelet transform (DWT) routines, not the DTCWT MATLAB toolbox.

Other baselines implement the methods cited in the manuscript, including
iTransformer, PatchTST, DLinear, FreTS, PAttn, and TSMixer. Please cite the
corresponding method papers when using their implementations or results.
