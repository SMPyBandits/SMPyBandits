{ hostPkgs ? import <nixpkgs> { }

, fetched ? s: (hostPkgs.nix-update-source.fetch s).src

, pkgs ? import (fetched ./pin.json) { }

}:
with pkgs.lib;
let

  SMPyBanditsPythonPackages = pkgs.python37Packages.override {
    overrides = self: super: rec {
      SMPyBandits = pkgs.callPackage ./pkgs/SMPyBandits {
        pythonPackages = self;
        src = let
          patterns = ''
            *
            !docs
            !SMPyBandits/**
            !SMPyBandits
            !LICENSE
            !setup.py
            !setup.cfg
            !requirements.txt../requirements_full.txt
            !MANIFEST.in
            !logo.png
            !logo_large.png
            !conf.py'';
        in pkgs.nix-gitignore.gitignoreSourcePure patterns ../.;
      };
    };
  };

  jupyterWithBatteries = pkgs.jupyter.override rec {
    python3 = SMPyBanditsPythonPackages.python.withPackages
      (ps: with ps; [ SMPyBanditsPythonPackages.SMPyBandits ipywidgets ]);
    definitions = {
      # This is the Python kernel we have defined above.
      python3 = {
        displayName = "Python 3";
        argv = [
          "${python3.interpreter}"
          "-m"
          "ipykernel_launcher"
          "-f"
          "{connection_file}"
        ];
        language = "python";
        logo32 = "${python3.sitePackages}/ipykernel/resources/logo-32x32.png";
        logo64 = "${python3.sitePackages}/ipykernel/resources/logo-64x64.png";
      };
    };
  };

in rec {

  pythonPackages = SMPyBanditsPythonPackages;

  experiment = pkgs.mkShell {
    name = "experiment";
    buildInputs = [
      jupyterWithBatteries
      (pythonPackages.python.withPackages
        (ps: with ps; [ SMPyBanditsPythonPackages.SMPyBandits ipywidgets ipython h5py]))
    ];
    EXAMPLE_NOTEBOOKS = ../notebooks;
  };
}
