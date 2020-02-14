{ src, fetchFromGitHub, pythonPackages }:
pythonPackages.buildPythonPackage {
  inherit src;
  name = "SMPYBandits";
  buildInputs = [ ];
  propagatedBuildInputs = with pythonPackages; [
    numpy
    numba
    scipy
    matplotlib
    seaborn
    tqdm
    scikitlearn
    scikit-optimize
  ];
  checkPhase = "true";
}
