# light-roast-test

Example scripts for running on the UChicago AF.

## How to run these examples

To run the eventloop example, use the `python 3.6` kernel.

To run the columnar example, you'll need a more recent version of python, and you probably want to work in a virtual environment.  Here are a few options for that:  

### Using pip

The simplest way to get a new environment up and running is with `pip`:

```
bash # in case you're not already in a bash shell
new_env = "my_new_virtual_env"
python -m venv ${new_env}
source ${new_env}/bin/activate
pip install ipykernel numpy dask_jobqueue parse
python -m ipykernel install --user --name=${new_env}
```

Then restart your jupyter server and you should see the new kernel available for use.  To add a module later, just open a terminal in jupyter:

```
bash # in case you're not already in a bash shell
new_env = "my_new_virtual_env"
source ${new_env}/bin/activate
pip install scikit-learn
```

No need to restart the jupyter kernel if you're just adding packages.


### Using pixi
In a Jupyter shell, set up a new kernel to use:

```
curl -fsSL https://pixi.sh/install.sh | bash
export PATH=$PATH:.pixi/bin/
pixi init
pixi add python=3.12
pixi add atlas-schema
pixi add ipykernel
pixi add pixi-kernel
pixi add numpy                  # this syntax should work in most cases, but maybe not all packages support this
pixi add pip                    # if you need pip
pip install dask_jobqueue parse # this should also work
python -m ipykernel install --user --name=light-roast-kernel
```

Then restart your Jupyter server.  On restart, you should see an option to use the `light-roast-kernel`, which is the one you want.  You may need to add some additional packages, which you can do from within a shell:

```
pixi shell
pip install parse               # this should also work
```

You may need to restart your Jupyter server (not just the kernel) again if you do that.

### Using ALRB

If you prefer to use ALRB, you can try something like the following, but it hasn't worked smoothly in the past:

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

The tool referenced [here](https://github.com/matthewfeickert/cvmfs-venv) may help with getting the environment right, but I'm not sure that's the problem.

