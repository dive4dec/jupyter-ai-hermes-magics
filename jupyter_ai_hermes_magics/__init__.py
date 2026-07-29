"""jupyter-ai-hermes-magics — %%hermes cell magic for Jupyter notebooks.

Talk to Hermes Agent directly inside a notebook cell.  Each ``%%hermes`` cell
gathers notebook context, sends the prompt to Hermes with full agent
capabilities (tools, memory, skills, self-learning), streams the response
into a transcript cell group below, and maintains a dot-notation session tree
for branching conversations.
"""

from .version import __version__
from .magics import HermesMagics, load_ipython_extension

__all__ = ["__version__"]
