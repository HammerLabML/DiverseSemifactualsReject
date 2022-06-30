# "Even if ..." -- Diverse Semifactual Explanations of Reject

This repository contains the implementation of the methods proposed in the paper ["Even if ..." -- Diverse Semifactual Explanations of Reject](paper.pdf) by André Artelt and Barbara Hammer.

The experiments as described in the paper are implemented in the folder [Implementation](Implementation/).

## Abstract

Machine learning based decision making systems applied in safety critical areas require reliable high certainty predictions. For this purpose, the system can be extended by an reject option which allows the system to reject inputs where only a prediction with an unacceptably low certainty would be possible. While being able to reject uncertain samples is important, it is also of importance to be able to explain why a particular sample was rejected. With the ongoing rise of eXplainable AI (XAI), a lot of explanation methodologies for machine learning based systems have been developed -- explaining reject options, however, is still a novel field where only very little prior work exists.

In this work, we propose to explain rejects by semifactual explanations, an instance of example-based explanation methods, which them self have not been widely considered in the XAI community yet. We propose a conceptual modeling of semifactual explanations for arbitrary reject options and empirically evaluate a specific implementation on a conformal prediction based reject option.

## Details
### Implementation of experiments
The shell script `run_experiments.sh` runs all experiments.

### Other (important) stuff
#### Computation of semifactual explanations

The implementation of the proposed algorithms for computing diverse semifactual explanations can be found in `Implementation/semifactual.py`.


## Requirements

- Python3.6
- Packages as listed in `Implementation/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE](LICENSE)

## How to cite

You can cite the version on [arXiv](TODO)
