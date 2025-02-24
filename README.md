# light-roast-test

Example scripts for running on the UChicago AF.

To run the eventloop example, use the `python 3.6` kernel.

To run the columnar example, you'll need a more recent version of python.  In a Jupyter shell, set up a new kernel to use:

```
curl -fsSL https://pixi.sh/install.sh | bash
export PATH=$PATH:.pixi/bin/
pixi init
pixi add python=3.12
pixi add atlas-schema
pixi add ipykernel
pixi add pixi-kernel
python -m ipykernel install --user --name=light-roast-kernel
```

Then restart your Jupyter server.  On restart, you should see an option to use the `light-roast-kernel`, which is the one you want.  You may need to add some additional packages, which you can do from within a shell:

```
pixi shell
pixi add numpy            # this syntax should work in most cases, but maybe not all packages support this
pixi add pip              # if you need pip
pip install dask_jobqueue # this should also work
```

You may need to restart Jupyter again if you do that, possibly.

If you prefer to use ALRB, you can try something like the following, but I don't find that it works for me:

```
export ATLAS_LOCAL_ROOT_BASE="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
alias setupATLAS="source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh"
setupATLAS
lsetup "python 3.11.9-x86_64-el9"
python3 -m venv lr-kernel
source lr-kernel/bin/activate
pip3 install --upgrade pip
pip3 install atlas-schema dask_jobqueue parse ipykernel
python3 -m ipykernel install --user --name=lr-kernel
```
