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

