
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

We highly encourage using a conda environment for the installation

* Install conda

  ```sh
  # Linux
  https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
  ```
  ```sh
  # Windows
  https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
  ```
* Install the conda environment with all the necessary dependencies
  
  Download the environment recipe for [Windows](https://github.com/gandresr/ptsnet/blob/development/conda/Windows/ptsnet.yml) or [Linux/Mac](https://github.com/gandresr/ptsnet/blob/development/conda/Linux/ptsnet.yml). Then `cd` to the location of the downloaded ptsnet.yml file. The parallel version of PTSNET can only run on Linux.
  
  ```sh
  # Open the shell and type
  conda env create -f ptsnet.yml
  conda activate ptsnet
  ```


<!-- USAGE EXAMPLES -->
## Usage

Use this [jupyter notebook](https://github.com/gandresr/ptsnet/blob/development/tests/test_functions.ipynb) to run useful examples of transient simulations using PTSNET. You can test the parallel version of PTSNET executing [test_simulation.py](https://github.com/gandresr/ptsnet/blob/development/tests/test_simulation.py) as follows:

```sh
mpiexec -n 8 python test_simulation.py
```

_For more examples, please refer to the [Documentation](https://example.com)_



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



<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Gerardo Riano - griano@utexas.edu

Project Link: [https://github.com/gandresr/PTSNET](https://github.com/gandresr/PTSNET)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

This publication was developed under Cooperative Agreement No. 83595001 awarded by the U.S. Environmental Protection Agency to The University of Texas at Austin. It has not been formally reviewed by EPA. The views expressed in this document are solely those of the authors, and do not necessarily reflect those of the Agency. EPA does not endorse any products or commercial services mentioned in this publication. The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing HPC resources that have contributed to the research results reported within this paper. URL: http://www.tacc.utexas.edu





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
