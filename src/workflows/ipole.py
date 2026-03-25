"""

  ipole.py (requires python3)

  provides 
   + hooks into running ipole 
   + (TODO) method to read (more of) ipole output to python dictionary

  2018.10.30 gnw

"""

import subprocess


def _run_command(cmd, verbose=0):
  if verbose > 1:
    completed = subprocess.run(cmd, check=True, text=True)
    return completed.stdout.splitlines() if completed.stdout else []

  completed = subprocess.run(
      cmd,
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
  )
  return completed.stdout.splitlines() if completed.stdout else []

def run(args, exe="./ipole", quench=False, unpol=False, parfile=None, verbose=0):
  """Runs ipole with config as specified by args."""

  cmd = [exe]
  if parfile is not None:
    cmd += ["-par",parfile]

  cmd += ["--{}={}".format(key,args[key]) for key in args]

  if quench: cmd += ["-quench"]
  if unpol: cmd += ["-unpol"]

  if verbose>0: print(" ".join(cmd))
  output = _run_command(cmd, verbose=verbose)

  results = {}
  for line in output:
    if "Ftot" in line:
      proc = line.replace('(','').replace(')','').split()
      results['Ftot_pol'] = float(proc[3])
      results['Ftot_unpol'] = float(proc[5])

  return results

def run_legacy(thetacam, freqcgs, Mbh, Munit, fname, Rlow=None, Rhigh=None, exe="./ipole", counterjet=0, 
        quench=False, verbose=False, unpol=False):
  """ runs ipole with config as specified by input arguments """
  if Rlow is None and Rhigh is not None: Rlow = 1.
  if Rhigh is None and Rlow is not None: Rhigh = 1.
  if Rlow is None:
    args = [ exe, thetacam, freqcgs, Mbh, Munit, fname, counterjet ]
  else:
    args = [ exe, thetacam, freqcgs, Mbh, Munit, fname, counterjet, Rlow, Rhigh ]
  args = [ str(x) for x in args ]
  if quench: args.append("-quench")
  if unpol: args.append("-unpol")
  if verbose: print(args)
  output = _run_command(args, verbose=2 if verbose else 0)
  results = {}
  for line in output:
    if "Ftot" in line:
      proc = line.replace('(','').replace(')','').split()
      results['Ftot_pol'] = float(proc[3])
      results['Ftot_unpol'] = float(proc[4])
  return results
