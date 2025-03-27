
<!-- [![Contributors][contributors-shield]][contributors-url] -->
<!-- [![Forks][forks-shield]][forks-url] -->
<!-- [![Stargazers][stars-shield]][stars-url] -->
<!-- [![Issues][issues-shield]][issues-url] -->
<!-- [![The Unlicense][license-shield]][license-url] -->
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/gandresr/PTSNET">
    <img src="https://github.com/gandresr/PTSNET/raw/development/docs/images/ptsnet_logo.png" alt="Logo" width="650" height="100">
  </a>


  <p align="center">
    Parallel Transient Simulation in Water Networks
    <br />
    <a href="https://github.com/gandresr/PTSNET"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/gandresr/PTSNET">View Demo</a>
    ·
    <a href="https://github.com/gandresr/PTSNET/issues">Report Bug</a>
    ·
    <a href="https://github.com/gandresr/PTSNET/issues">Request Feature</a>
  </p>
</p>

[![Downloads](https://static.pepy.tech/personalized-badge/ptsnet?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/ptsnet)
<!-- TABLE OF CONTENTS -->
<!-- <details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details> -->



<!-- ABOUT THE PROJECT -->
<!-- ## About The Project -->

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

<!-- Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`gandresr`, `PTSNET`, `twitter_handle`, `email`, `project_title`, `project_description`
 -->

<!-- ### Built With -->

<!-- * []()
* []()
* []() -->



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps. PTSNET can be downloaded via pip

### Installation

We highly encourage using a conda environment for the installation, so that dependencies such as OpenMPI don't have to be manually installed.

* Install conda

  ```sh
  # Linux
  https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
  ```
  ```sh
  # Windows
  https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
  ```
* Install the conda environment with all the necessary dependencies, by opening a terminal and running the following commands
  1. Download [environment.yml](https://raw.githubusercontent.com/gandresr/ptsnet/refs/heads/development/environment.yml)
  2. Execute `conda activate` to start conda
  3. Execute `conda env create -f environment.yml`
  4. Execute `conda activate ptsnet`

<!-- USAGE EXAMPLES -->
## Usage

Create a file called named `simulation.py` with the following contents:

```python
import matplotlib.pyplot as plt
from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path

sim = PTSNETSimulation(
  workspace_name = 'TNET3_VALVE',
  inpfile = get_example_path('TNET3'))
sim.define_valve_operation('VALVE-179', initial_setting=1, final_setting=0, start_time=1, end_time=2)
sim.run(); print(sim)

plt.plot(sim['time'], sim['node'].head['JUNCTION-73'])
plt.show()
```

After creating the file, you can execute the code from the command line.

#### To execute the parallel version of PTSNET it is necessary to have __Linux/Mac__</span>. 

If you have __Linux/Mac__ execute the following command on the terminal:
```sh
mpiexec -n 4 python simulation.py
```
The number of processors is defined by the parameter `-n` in the command, in this case 4.

If you have __Windows__ you can still run the simulation as shown below, but you will not have access to PTSNET's parallel capabilities:
```sh
python simulation.py
```
For more examples, please refer to the [jupyter notebooks](https://github.com/gandresr/ptsnet/tree/development/publication).



<!-- ROADMAP -->
<!-- ## Roadmap -->

<!-- See the [open issues](https://github.com/gandresr/PTSNET/issues) for a list of proposed features (and known issues). -->



<!-- CONTRIBUTING -->
<!-- ## Contributing -->

<!-- Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**. -->

<!-- 1. Fork the Project -->
<!-- 2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`) -->
<!-- 3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`) -->
<!-- 4. Push to the Branch (`git push origin feature/AmazingFeature`) -->
<!-- 5. Open a Pull Request -->


<!-- Cite Us -->
## Cite Us

If the PTSNET package has been useful for your research, please cite the paper below:

[PTSNet: A Parallel Transient Simulator for Water Transport Networks based on vectorization and distributed computing](https://www.sciencedirect.com/science/article/pii/S1364815222002547)
```
@article{riano2022ptsnet,
  title={PTSNet: A Parallel Transient Simulator for Water Transport Networks based on vectorization and distributed computing},
  author={Ria{\~n}o-Brice{\~n}o, Gerardo and Hodges, Ben R and Sela, Lina},
  journal={Environmental Modelling \& Software},
  volume={158},
  pages={105554},
  year={2022},
  publisher={Elsevier}
}
```

If the PTSNET algorithm was useful for you, please cite the paper below:

[Distributed and Vectorized Method of Characteristics for Fast Transient Simulations in Water Distribution Systems](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.12709)
```
@article{riano2022distributed,
  title={Distributed and vectorized method of characteristics for fast transient simulations in water distribution systems},
  author={Ria{\~n}o-Brice{\~n}o, Gerardo and Sela, Lina and Hodges, Ben R},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  year={2022},
  publisher={Wiley Online Library}
}
```

<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Gerardo Riano - griano@utexas.edu

Lina Sela - linasela@utexas.edu

Project Link: [https://github.com/gandresr/PTSNET](https://github.com/gandresr/PTSNET)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing HPC resources that have contributed to the research results reported within this publication. This work was supported in part by NSF under award 2015658 and Cooperative Agreement No. 83595001 awarded by the U.S. Environmental Protection Agency to The University of Texas at Austin. It has not been formally reviewed by EPA. The views expressed in this presentation are solely those of the authors, and do not necessarily reflect those of the Agency. EPA does not endorse any products or commercial services mentioned in this publication.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/gandresr/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/gandresr/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/gandresr/repo.svg?style=for-the-badge
[forks-url]: https://github.com/gandresr/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/gandresr/repo.svg?style=for-the-badge
[stars-url]: https://github.com/gandresr/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/gandresr/repo.svg?style=for-the-badge
[issues-url]: https://github.com/gandresr/PTSNET/issues
[license-shield]: https://img.shields.io/github/license/gandresr/repo.svg?style=for-the-badge
[license-url]: https://github.com/gandresr/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/gandresr
