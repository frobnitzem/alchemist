from pathlib import Path
from collections.abc import AsyncIterator
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import asyncio
import os
import logging
_logger = logging.getLogger(__name__)

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import numpy as np
import typer
app = typer.Typer()

# raw async stream.py methods
async def take(s, n, progress=False):
    i = 0
    if progress:
        p = tqdm(total=n)
    async for x in s:
        if i >= n:
            break
        yield x
        i += 1
        if progress:
            p.update(1)

async def to_async(g):
    for x in g:
        yield x

async def consume(stream):
    async for i in stream:
        pass

async def apply(it, fn):
    async for z in it:
        yield fn(*z)

@app.command()
def train(config: Annotated[Path, typer.Argument(help="TrainConfig yaml file.")],
          inp:    Annotated[Optional[Path], typer.Option(help="Starting model checkpoint path")] = None,
          out:    Annotated[Path, typer.Option(help="Output checkpoint path")] = "model.pth",
         ):
    """ Train (or continue training of) a neural network
        model for generating structures.
    """

    #trainConfig = TrainConfig.model_validate(load_dict(config))
    #with Parallel(world_size=os.environ.get("SLURM_NTASKS", None),
    #              world_rank=os.environ.get("SLURM_PROCID", None),
    #              local_rank=os.environ.get("SLURM_LOCALID", None),
    #              num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK", None)) \
    #        as parallel:
    #    if inp is not None:
    #        ann = GenNN.load(inp, parallel)
    #    else:
    #        ann = GenNN(parallel, trainConfig)
    #    ann.train(trainConfig.dataset, trainConfig.training, out)
    print("not implemented")

@app.command()
def generate(inp: Annotated[Path, typer.Argument(help="Model checkpoint path")],
             db: Annotated[Path, typer.Argument(help="Input structures for generator.")],
             out: Annotated[Path, typer.Argument(help="ASE DB to store results.")],
             s: Annotated[int, typer.Option(help="Number of structures to generate.")] = 0):
    """ Use a trained model to generate structures.
    """
    #with Parallel(world_size=os.environ.get("SLURM_NTASKS", None),
    #              world_rank=os.environ.get("SLURM_PROCID", None),
    #              local_rank=os.environ.get("SLURM_LOCALID", None),
    #              num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK", None)) \
    #        as parallel:
    #    assert parallel.world_size == 1, "Need multiple dbs for parallel operation."

    #    ann = GenNN.load(inp, parallel)

    #    g = apply(iter_mols(db), ann.generate)
    #    if s > 0:
    #        g = take(g, s, True)
    #    else:
    #        g = atqdm(g)

    #    asyncio.run(
    #        consume(
    #            add_mols(g, out)
    #        )
    #    )
    print("not implemented")

@app.command()
def dynamics(config: Annotated[Path, typer.Argument(help="DynamicConfig parameters.")],
             s: Annotated[int, typer.Argument(help="Number of structures to generate.")],
             out: Annotated[Path, typer.Argument(help="Output file.")]
            ):
    """ Run dynamics on a model to generate structures.
    """
    print("not implemented")

    ##config = DynamicConfig.model_validate(**load_dict(config))
    #config = DynamicConfig.model_validate(load_dict(config))
    ##data = LJDataset(config, s, out)
    ##data.process()

    #volume = config.nparticles*(config.sigma**3)/config.reduced_density
    #box_edge = volume**(1.0/3.0)

    #names = ['Ar']*config.nparticles
    #cell  = np.eye(3)*box_edge
    #def get_ase(x, e):
    #    return to_ase(names, x, cell, energy=e)

    #asyncio.run(
    #        consume(
    #            take(
    #                add_mols(
    #                    apply(
    #                        to_async(simulate_system(config)),
    #                        get_ase),
    #                    out),
    #            s, True)
    #        )
    #    )
