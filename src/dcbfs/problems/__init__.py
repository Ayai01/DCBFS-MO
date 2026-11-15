"""Problem registry: mathematical and domain benchmarks share one interface."""
from functools import partial
from ..utils.numba_cache import register_native_cache

register_native_cache()

from .classic import CLASSIC, Classic


from .landscapes import LANDSCAPE_NAMES, LandscapeProblem
from .panda import PandaTrajectory


def _parallel_problem(factory, **options):
    problem = factory(**options)
    problem.parallel_evaluation = True
    return problem


def _panda_problem(name,**options):
    query='cage-0001' if name=='panda' else name[len('panda-'):]
    if options.pop('query',query)!=query:
        raise ValueError('Use the registered panda-<scene>-<query> name for a different task')
    problem=_parallel_problem(PandaTrajectory,query=query,**options)
    problem.name=name
    return problem


PROBLEMS = {name: partial(Classic, name) for name in CLASSIC}

PROBLEMS['panda']=partial(_panda_problem,'panda')
for _scene in ('cage','bookshelf_tall'):
    for _number in range(1,6):
        _name=f'panda-{_scene}-{_number:04}'
        PROBLEMS[_name]=partial(_panda_problem,_name)

PROBLEMS.update({name: partial(LandscapeProblem,name) for name in LANDSCAPE_NAMES})


def make_problem(name, **options):
    """Construct a registered batch problem from its configuration options."""
    name = name.lower()
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem {name!r}; choose {', '.join(PROBLEMS)}")
    return PROBLEMS[name](**options)


def analytic_reference(definition):
    if definition["generator"]=="landscape-analytic":
        from .landscapes import analytic_reference as generate
    else:
        from .classic import analytic_reference as generate
    return generate(definition)



PROBLEMS={key:value for key,value in PROBLEMS.items()if key in ['c2dtlz2', 'c3dtlz4', 'disconnected-corridor', 'dtlz2', 'dtlz7', 'narrow-corridor', 'panda-bookshelf_tall-0001', 'panda-bookshelf_tall-0002', 'panda-bookshelf_tall-0003', 'panda-bookshelf_tall-0004', 'panda-bookshelf_tall-0005', 'panda-cage-0001', 'panda-cage-0002', 'panda-cage-0003', 'panda-cage-0004', 'panda-cage-0005', 'wfg2', 'wfg4', 'wfg9']}
